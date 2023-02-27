#https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1612
#https://github.com/padmalcom/wav2vec2-nonverbalvocalization/blob/main/Wav2Vec2ClassificationHead.py#L4


import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from torch import nn

_HIDDEN_STATES_START_POSITION = 2


class Wav2Vec2ForCTCnCLS(Wav2Vec2PreTrainedModel):

    def __init__(self, config, accent=None, domain=None, vad=None,
                 accent_len=72, domain_len=3, vad_len=2, alphas="0.1|0.3|0.6",
                 loss_reduction="sum"):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.accent = accent
        self.domain = domain
        self.vad = vad
        self.loss_reduction = loss_reduction
        if self.accent:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.accent_head = nn.Linear(config.hidden_size, accent_len)
        if self.domain:
            self.domain_head = nn.Linear(config.hidden_size, domain_len)
        if self.vad:
            self.vad_head = nn.Linear(config.hidden_size, vad_len)
        self.init_weights()
        self.alphas = [float(alpha) for alpha in alphas.split("|")]

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
       
    def freeze_feature_encoder(self):
        self.freeze_feature_extractor()

    def _ctc_loss(self, logits, labels, input_values, attention_mask=None):
        loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            # log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        return loss

    def _accent_loss(self, logits, accent_labels):
        # sum hidden_states over dim 1 (the sequence length), then feed into self.accent
        loss = None
        if accent_labels is not None:
            # nn.BCEWithLogitsLoss()
            loss = F.cross_entropy(logits, accent_labels.to(logits.device))
        return loss

    def _domain_loss(self, logits, domain_labels):
        # sum hidden_states over dim 1 (the sequence length), then feed into self.domain_head
        loss = None
        if domain_labels is not None:
            loss = F.cross_entropy(logits, domain_labels.to(logits.device))
        return loss

    def _vad_loss(self, logits, vad_labels):
        # sum hidden_states over dim 1 (the sequence length), then feed into self.domain_head
        loss = None
        if vad_labels is not None:
            loss = F.cross_entropy(logits, vad_labels.to(logits.device))
        return loss

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,  # tuple: (ctc_labels, accent_labels), shape=(batch_size, target_length)
            accent=True,
            domain=True,
            vad=True
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        print("input_values", input_values.shape, input_values)
        print("labels", labels.shape, labels)
        print("accent", accent['input_ids'].shape, accent['input_ids'])
        print("domain", domain, self.domain)
        print("vad", vad, self.vad)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # this is the last layer's hidden states
        hidden_states = self.dropout(hidden_states)
        logits_accent = logits_domain = logits_vad = None

        # head 1
        logits_ctc = self.lm_head(hidden_states)

        if self.accent:
            x = self.dense(hidden_states)
            x = torch.tanh(x)
            x = self.dropout(x)
            logits_accent = self.accent_head(x)

        if self.domain:
            logits_domain = self.domain_head(hidden_states)

        if self.vad:
            logits_vad = self.vad_head(hidden_states)

        loss = None
        task_losses = []
        if labels is not None:
            loss_ctc = self._ctc_loss(logits_ctc, labels, input_values, attention_mask)
            print("loss_ctc: ", loss_ctc)
            task_losses.append(loss_ctc)
            if self.accent:
                accent = accent['input_ids']
                loss_accent = self._accent_loss(logits_accent, accent)
                print("loss_accent: ", loss_accent)
                task_losses.append(loss_accent)
            if self.domain:
                domain = domain['input_ids']
                loss_domain = self._domain_loss(logits_domain, domain)
                task_losses.append(loss_domain)
            if self.vad:
                vad = vad['input_ids']
                loss_vad = self._vad_loss(logits_vad, vad)
                task_losses.append(loss_vad)
        
        loss = self.compute_loss_reduction(task_losses, mode=self.loss_reduction)
        print("loss_reduced: ", type(loss), loss)

        if not return_dict:
            output = (logits_ctc, ) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits_ctc,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    
    def compute_loss_reduction(self, task_losses, mode="sum"):
        if mode == "sum":
            return torch.sum(task_losses)
        if mode == "mean":
            return torch.mean(task_losses)
        if mode == "weighted":
            assert len(task_losses) == len(self.alphas)
            return torch.sum(torch.tensor([loss * self.alphas[i] for i, loss in enumerate(task_losses)]))
        raise NotImplementedError
