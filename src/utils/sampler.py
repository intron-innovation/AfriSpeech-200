from typing import Any, Dict, Iterator, List, Optional, Union
import time 
import math
import torch
from torch.utils.data import IterableDataset, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import datasets
from datasets import load_dataset, Dataset
import transformers
from transformers import Trainer, Seq2SeqTrainer, set_seed, is_apex_available
from transformers.utils import is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import get_last_checkpoint, is_main_process, has_length
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
)

if is_apex_available():
    from apex import amp
    
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=None, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:
    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch
    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use torch for the random part as a distributed sampler 
    # will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = mega_batch_mult * batch_size 
    # 50 minibatches per megabatch, mgbatch_size is num examples
    megabatches = [indices[i : i + megabatch_size].tolist() 
                   for i in range(0, len(lengths), megabatch_size)]
    megabatches = [list(sorted(megabatch, key=lambda i: lengths[i], reverse=True)) 
                   for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]


class IntronLengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together 
    features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        lengths: List[int],
        generator=None,
    ):
        self.batch_size = batch_size
        if lengths is None:
            raise ValueError("lengths cannot be none")
        if isinstance(lengths, list):
            pass
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. "\
                "Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        print("sorted indices: ", indices[:10])
        return iter(indices)
    
    def _get_indices(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        return indices


class IntronDistributedLengthGroupedSampler(DistributedSampler):
    r"""
    Distributed Sampler that samples indices in a way that groups together features of the dataset of roughly the same
    length while keeping a bit of randomness.
    """
    # Copied and adapted from PyTorch DistributedSampler.
    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        lengths: List[int] = None,
        model_input_name: Optional[str] = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, DistributedLengthGroupedSampler will be slow. Converting lengths to"
                " List[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.lengths) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil((len(self.lengths) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=g)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        print("sorted indices: ", indices[:10])
        return iter(indices)

    
class IntronSeq2SeqTrainer(Seq2SeqTrainer):
    
    def __init__(self, sampler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler
    
    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        # Deprecated code
        if self.args.use_legacy_prediction_loop:
            if is_torch_tpu_available():
                return SequentialDistributedSampler(
                    eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
                )
            elif is_sagemaker_mp_enabled():
                return SequentialDistributedSampler(
                    eval_dataset,
                    num_replicas=smp.dp_size(),
                    rank=smp.dp_rank(),
                    batch_size=self.args.per_device_eval_batch_size,
                )
            elif self.args.local_rank != -1:
                return SequentialDistributedSampler(eval_dataset)
            else:
                return SequentialSampler(eval_dataset)

        # Build the sampler.
        if self.args.group_by_length:
            generator, seed = self._get_generator_and_seed()
            if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
                lengths = (
                    eval_dataset[self.args.length_column_name]
                    if self.args.length_column_name in eval_dataset.column_names
                    else None
                )
            elif is_datasets_available() and isinstance(eval_dataset, torch.utils.data.Dataset):
                lengths = (
                    eval_dataset.asr_data[self.args.length_column_name]
                    if self.args.length_column_name in eval_dataset.asr_data.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return IntronLengthGroupedSampler(
                    self.args.per_device_eval_batch_size,
                    lengths=lengths,
                    generator=generator,
                )
            else:
                return IntronDistributedLengthGroupedSampler(
                    self.args.per_device_eval_batch_size,
                    dataset=eval_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return ShardSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )
    
    def _get_generator_and_seed(self):
        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here 
            # (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return generator, seed
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        elif self.args.group_by_length:
            if self.train_dataset is None or not has_length(self.train_dataset):
                return None

            generator, seed = self._get_generator_and_seed()
            
            # Build the sampler.
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            elif is_datasets_available() and isinstance(self.train_dataset, torch.utils.data.Dataset):
                lengths = (
                    self.train_dataset.asr_data[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.asr_data.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return IntronLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    lengths=lengths,
                    generator=generator,
                )
            else:
                return IntronDistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.sortish_sampler:
                self.train_dataset.make_sortish_sampler(
                    self.args.per_device_train_batch_size,
                    distributed=(self.args.parallel_mode == ParallelMode.DISTRIBUTED),
                )
            elif self.args.local_rank == -1:
                if self.sampler == 'random':
                    return RandomSampler(self.train_dataset, generator=generator)
                else:
                    return SequentialSampler(self.train_dataset)
            return (
                DistributedSampler(self.train_dataset)
            ) 

    
class IntronTrainer(Trainer):
    
    def __init__(self, sampler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler
        
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.args.device)
                if self.deepspeed and inputs[k].dtype != torch.int64:
                    kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
                inputs[k] = v.to(**kwargs)

            if k == 'labels': # labels are list of tensor, not tensor, special handle here
                for i in range(len(inputs[k])):
                   
                    kwargs = dict(device=self.args.device)
                    if self.deepspeed and inputs[k][i].dtype != torch.int64:
                        kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
                    inputs[k][i] = inputs[k][i].to(**kwargs)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()
    
    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        # Deprecated code
        if self.args.use_legacy_prediction_loop:
            if is_torch_tpu_available():
                return SequentialDistributedSampler(
                    eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
                )
            elif is_sagemaker_mp_enabled():
                return SequentialDistributedSampler(
                    eval_dataset,
                    num_replicas=smp.dp_size(),
                    rank=smp.dp_rank(),
                    batch_size=self.args.per_device_eval_batch_size,
                )
            elif self.args.local_rank != -1:
                return SequentialDistributedSampler(eval_dataset)
            else:
                return SequentialSampler(eval_dataset)

        # Build the sampler.
        if self.args.group_by_length:
            generator, seed = self._get_generator_and_seed()
            if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
                lengths = (
                    eval_dataset[self.args.length_column_name]
                    if self.args.length_column_name in eval_dataset.column_names
                    else None
                )
            elif is_datasets_available() and isinstance(eval_dataset, torch.utils.data.Dataset):
                lengths = (
                    eval_dataset.asr_data[self.args.length_column_name]
                    if self.args.length_column_name in eval_dataset.asr_data.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return IntronLengthGroupedSampler(
                    self.args.per_device_eval_batch_size,
                    lengths=lengths,
                    generator=generator,
                )
            else:
                return IntronDistributedLengthGroupedSampler(
                    self.args.per_device_eval_batch_size,
                    dataset=eval_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return ShardSampler(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )
    
    def _get_generator_and_seed(self):
        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here 
            # (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return generator, seed
        
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator, seed = self._get_generator_and_seed()

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            elif is_datasets_available() and isinstance(self.train_dataset, torch.utils.data.Dataset):
                lengths = (
                    self.train_dataset.asr_data[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.asr_data.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return IntronLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    lengths=lengths,
                    generator=generator,
                )
            else:
                return IntronDistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                if self.sampler == 'random':
                    return RandomSampler(self.train_dataset, generator=generator)
                else:
                    return SequentialSampler(self.train_dataset)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
