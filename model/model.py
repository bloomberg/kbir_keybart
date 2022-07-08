import logging
from transformers import PreTrainedModel, RobertaConfig, RobertaModel
from transformers.activations import gelu, ACT2FN
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaLMHead,
)
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCELoss
from klm.model.outputs import KLMForReplacementAndMaskedLMOutput
from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, TokenClassifierOutput
from transformers.models.encoder_decoder.configuration_encoder_decoder import (
    EncoderDecoderConfig,
)

logger = logging.getLogger(__name__)


class ReplacementClassificationHead(nn.Module):
    def __init__(self, config, use_doc_emb=True):
        super(ReplacementClassificationHead, self).__init__()
        self.use_doc_emb = use_doc_emb
        classifier_hidden_size = 2 * config.hidden_size
        if self.use_doc_emb:
            classifier_hidden_size += config.hidden_size
        self.num_labels = 2
        self.classifier = nn.Linear(classifier_hidden_size, self.num_labels)
        self.bias = nn.Parameter(torch.zeros(self.num_labels))

    def forward(self, hidden_states, pooled_states, pairs):
        bs, num_pairs, _ = pairs.size()
        bs, seq_len, dim = hidden_states.size()
        bs, dim = pooled_states.size()
        # pair indices: (bs, num_pairs)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        # (bs, num_pairs, dim)
        left_hidden = torch.gather(
            hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim)
        )
        # bs * num_pairs, dim
        left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim)
        # (bs, num_pairs, dim)
        right_hidden = torch.gather(
            hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim)
        )
        # bs * num_pairs, dim
        right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim)
        # bs * num_pairs, 2*dim
        hidden_states = torch.cat((left_hidden, right_hidden), -1)

        if self.use_doc_emb:
            # bs * num_pairs, dim
            pooled_states = (
                pooled_states.unsqueeze(1)
                .repeat(1, num_pairs, 1)
                .view(bs * num_pairs, dim)
            )
            hidden_states = torch.cat((pooled_states, hidden_states), -1)

        # target scores : bs * num_pairs, num_labels
        target_scores = self.classifier(hidden_states) + self.bias
        target_scores = torch.reshape(target_scores, (bs, num_pairs, self.num_labels))
        return target_scores


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class MLPWithLayerNorm(nn.Module):
    def __init__(self, config, input_size):
        super(MLPWithLayerNorm, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(input_size, config.hidden_size)
        self.non_lin1 = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )
        self.layer_norm1 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.non_lin2 = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )
        self.layer_norm2 = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden):
        return self.layer_norm2(
            self.non_lin2(
                self.linear2(self.layer_norm1(self.non_lin1(self.linear1(hidden))))
            )
        )


class InfillingHead(nn.Module):
    def __init__(
        self,
        config,
        roberta_model_embedding_weights,
        kp_max_seq_len=10,
        position_embedding_size=200,
    ):
        super(InfillingHead, self).__init__()
        classifier_hidden_size = 2 * config.hidden_size
        self.num_labels = kp_max_seq_len
        self.num_tok_classifier = nn.Linear(classifier_hidden_size, self.num_labels)
        self.bias = nn.Parameter(torch.zeros(self.num_labels))
        self.position_embeddings = nn.Embedding(kp_max_seq_len, position_embedding_size)
        self.mlp_layer_norm = MLPWithLayerNorm(
            config, config.hidden_size * 2 + position_embedding_size
        )
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            roberta_model_embedding_weights.size(1),
            roberta_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = roberta_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(roberta_model_embedding_weights.size(0)))
        self.kp_max_seq_len = kp_max_seq_len

    def forward(self, hidden_states, pairs):
        bs, num_pairs, _ = pairs.size()
        bs, seq_len, dim = hidden_states.size()
        # pair indices: (bs, num_pairs)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        # (bs, num_pairs, dim)
        left_hidden = torch.gather(
            hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim)
        )
        # pair states: bs * num_pairs, kp_max_seq_len, dim
        kp_left_hidden = (
            left_hidden.contiguous()
            .view(bs * num_pairs, dim)
            .unsqueeze(1)
            .repeat(1, self.kp_max_seq_len, 1)
        )
        # bs * num_pairs, dim
        num_tok_left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim)
        # (bs, num_pairs, dim)
        right_hidden = torch.gather(
            hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim)
        )
        # pair states: bs * num_pairs, kp_max_seq_len, dim
        kp_right_hidden = (
            right_hidden.contiguous()
            .view(bs * num_pairs, dim)
            .unsqueeze(1)
            .repeat(1, self.kp_max_seq_len, 1)
        )
        # bs * num_pairs, dim
        num_tok_right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim)
        # bs * num_pairs, 2*dim
        hidden_states = torch.cat((num_tok_left_hidden, num_tok_right_hidden), -1)

        # target scores : bs * num_pairs, num_labels
        num_tok_scores = self.num_tok_classifier(hidden_states)
        num_tok_scores = torch.reshape(num_tok_scores, (bs, num_pairs, self.num_labels))

        # (max_targets, dim)
        position_embeddings = self.position_embeddings.weight
        hidden_states = self.mlp_layer_norm(
            torch.cat(
                (
                    kp_left_hidden,
                    kp_right_hidden,
                    position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1),
                ),
                -1,
            )
        )
        # target scores : bs * num_pairs, kp_max_seq_len, vocab_size
        kp_logits = self.decoder(hidden_states) + self.bias
        return kp_logits, num_tok_scores


class KLMForReplacementAndMaskedLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = []

    def __init__(
        self,
        config,
        use_doc_emb=False,
        kp_max_seq_len=10,
        mlm_loss_weight=1.0,
        replacement_loss_weight=1.0,
        keyphrase_infill_loss_weight=1.0,
        infill_num_tok_loss_weight=1.0,
    ):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()
        self.replacement_classification_head = ReplacementClassificationHead(
            config, use_doc_emb
        )
        self.infilling_head = InfillingHead(
            config, self.roberta.embeddings.word_embeddings.weight, kp_max_seq_len
        )
        self.mlm_loss_weight = mlm_loss_weight
        self.replacement_loss_weight = replacement_loss_weight
        self.keyphrase_infill_loss_weight = keyphrase_infill_loss_weight
        self.infill_num_tok_loss_weight = infill_num_tok_loss_weight
        self.kp_max_seq_len = kp_max_seq_len

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        keyphrases_input_ids=None,
        keyphrase_pairs=None,
        replacement_labels=None,
        masked_keyphrase_pairs=None,
        masked_keyphrase_labels=None,
        keyphrase_mask_num_tok_labels=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        prediction_scores = self.lm_head(sequence_output)
        pooled_output = outputs[1]

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )
            masked_lm_loss = self.mlm_loss_weight * masked_lm_loss

        replacement_logits = None
        if keyphrase_pairs is not None and replacement_labels is not None:
            replacement_logits = self.replacement_classification_head(
                sequence_output, pooled_output, keyphrase_pairs
            )
            if replacement_labels is not None:
                loss_fct = CrossEntropyLoss()
                # As this is a binary classification num_classes is fixed at 2
                num_class = 2
                replacement_classification_loss = loss_fct(
                    replacement_logits.view(-1, num_class), replacement_labels.view(-1)
                )
                masked_lm_loss += (
                    self.replacement_loss_weight * replacement_classification_loss
                )

        if (
            masked_keyphrase_pairs is not None
            and masked_keyphrase_labels is not None
            and keyphrase_mask_num_tok_labels is not None
        ):
            label_logits, num_toks_logits = self.infilling_head(
                sequence_output, masked_keyphrase_pairs
            )
            loss_fct = CrossEntropyLoss()
            masked_keyphrase_loss = loss_fct(
                label_logits.view(-1, self.config.vocab_size),
                masked_keyphrase_labels.view(-1),
            )
            masked_lm_loss += self.keyphrase_infill_loss_weight * masked_keyphrase_loss
            num_tok_loss_fct = CrossEntropyLoss()
            num_tok_loss = num_tok_loss_fct(
                num_toks_logits.view(-1, self.kp_max_seq_len),
                keyphrase_mask_num_tok_labels.view(-1),
            )
            masked_lm_loss += self.infill_num_tok_loss_weight * num_tok_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return KLMForReplacementAndMaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            replacement_logits=replacement_logits,
        )
