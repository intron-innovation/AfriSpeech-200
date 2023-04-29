# https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1612
# https://github.com/padmalcom/wav2vec2-nonverbalvocalization/blob/main/Wav2Vec2ClassificationHead.py#L4
# https://github.com/padmalcom/wav2vec2-nonverbalvocalization/blob/main/Wav2Vec2ForSpeechClassification.py
# https://www.v7labs.com/blog/multi-task-learning-guide
# https://medium.com/@mrityu.jha/understanding-the-grad-of-autograd-fc8d266fd6cf
# class weights https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from torch import nn

_HIDDEN_STATES_START_POSITION = 2


class Wav2Vec2ForCTCnCLS(Wav2Vec2PreTrainedModel):

    def __init__(self, config, accent=True, domain=True, vad=False,
                 accent_len=80, domain_len=3, vad_len=2, 
                 alphas="asr-0.7|accent-0.2|domain-0.1",
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
        if self.loss_reduction != "weighted":
            alphas="asr-0.7|accent-0.2|domain-0.1"
        self.alphas = {alpha.split('-')[0]: float(alpha.split('-')[1]) for alpha in alphas.split("|")}
        self.num_tasks = (accent + domain + vad + 1)
        if self.loss_reduction == "weighted":
            assert len(self.alphas) == self.num_tasks
            assert round(sum(list(self.alphas.values())), 4) == 1.0
        print("loss config", self.loss_reduction, self.alphas)

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
            # print("ctc logits", logits.shape, labels.shape)
            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            # print("ctc log_probs", log_probs.shape)

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
            # print("accent logits", logits.shape, accent_labels.shape)
            loss = F.cross_entropy(logits, accent_labels.to(logits.device))
        return loss

    def _domain_loss(self, logits, domain_labels):
        # sum hidden_states over dim 1 (the sequence length), then feed into self.domain_head
        loss = None
        if domain_labels is not None:
            # print("domain logits", logits.shape, domain_labels.shape)
            loss = F.cross_entropy(logits, domain_labels.to(logits.device))
        return loss

    def _vad_loss(self, logits, vad_labels):
        # sum hidden_states over dim 1 (the sequence length), then feed into self.domain_head
        loss = None
        if vad_labels is not None:
            loss = F.binary_cross_entropy(logits, vad_labels.to(logits.device))
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

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # this is the last layer's hidden states
        # print("hidden_states", hidden_states.shape)
        
        hidden_states = self.dropout(hidden_states)
        logits_accent = logits_domain = logits_vad = None

        # head 1
        logits_ctc = self.lm_head(hidden_states)

        # mean, max, sum pooling
        hidden_states = torch.mean(hidden_states, dim=1)
        
        if self.accent:
            accent_x1 = self.dense(hidden_states)  #####
            x = torch.tanh(accent_x1)
            x = self.dropout(x)
            logits_accent = self.accent_head(x)

        if self.domain:
            logits_domain = self.domain_head(hidden_states)

        if self.vad:
            logits_vad = self.vad_head(hidden_states)
  
        loss = None
        if labels is not None:
            loss_ctc = self._ctc_loss(logits_ctc, labels, input_values, attention_mask)
            # print("loss_ctc", loss_ctc)
            loss = loss_ctc * self.alphas['asr']
            num_losses = torch.tensor(1)
            if self.accent:
                accent = accent['input_ids']
                loss_accent = self._accent_loss(logits_accent, accent)
                # print("loss_accent", loss_accent)
                loss += (loss_accent * self.alphas['accent'])
                num_losses += 1
            if self.domain:
                domain = domain['input_ids']
                loss_domain = self._domain_loss(logits_domain, domain)
                # print("loss_domain", loss_domain)
                loss += (loss_domain * self.alphas['domain'])
                num_losses += 1
            if self.vad:
                vad = vad['input_ids']
                loss_vad = self._vad_loss(logits_vad, vad)
                loss += (loss_vad * self.alphas['vad'])
                num_losses += 1
         
            loss = self.compute_loss_reduction(loss, num_losses, mode=self.loss_reduction)
            # print("loss_reduced: ", type(loss), loss)

        if not return_dict:
            print("e dey work")
            output = (logits_ctc, ) + outputs[_HIDDEN_STATES_START_POSITION:]  
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=(logits_ctc, accent_x1),
            hidden_states=outputs.hidden_states, attentions=outputs.attentions, 
        )
    
    def compute_loss_reduction(self, task_loss, num_losses, mode="sum"):
        if mode == "sum" or mode == "weighted":
            return task_loss
        if mode == "mean":
            return task_loss / num_losses
        raise NotImplementedError
