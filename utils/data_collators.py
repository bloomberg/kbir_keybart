import logging
import time
import torch
import random
from dataclasses import dataclass
from typing import Dict
import time

from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import _collate_batch, tolist
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger()


@dataclass
class DataCollatorForKLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tokenizer,
        mlm_probability=0.15,
        kp_mask_percentage=0.4,
        kp_replace_percentage=0.4,
        keyphrase_universe_ids=None,
        max_keyphrase_pairs=20,
        max_seq_len=512,
        label_ignore_index=-100,
        do_generation=False,
        use_bart=False,
        do_keyphrase_generation=False,
        do_keyphrase_infilling=False,
        kp_max_seq_len=10,
        max_mask_keyphrase_pairs=10,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.kp_mask_percentage = kp_mask_percentage
        self.kp_replace_percentage = kp_replace_percentage
        self.keyphrase_universe_ids = keyphrase_universe_ids
        self.max_keyphrase_pairs = max_keyphrase_pairs
        self.max_mask_keyphrase_pairs = max_mask_keyphrase_pairs
        self.max_seq_len = max_seq_len
        self.kp_max_seq_len = kp_max_seq_len
        self.label_ignore_index = label_ignore_index
        self.pad_start_index = 1
        self.pad_end_index = 0
        self.do_generation = do_generation
        self.use_bart = use_bart
        self.do_keyphrase_generation = do_keyphrase_generation
        self.do_keyphrase_infilling = do_keyphrase_infilling
        if self.do_keyphrase_infilling:
            self.kp_mask_percentage = 0.0
            self.kp_infill_percentage = kp_mask_percentage
        else:
            self.kp_infill_percentage = 0.0

    def __call__(self, examples):
        original_input_ids = []
        updated_input_ids = []
        mask_labels = []
        kp_mask_labels = []
        overall_keyphrase_indexes = []
        overall_keyphrase_replacement_labels = []
        overall_catseq_keyphrase_input_ids = []
        overall_catseq_keyphrase_decoder_input_ids = []
        overall_masked_keyphrase_indexes = []
        overall_masked_keyphrase_labels = []
        overall_keyphrase_mask_num_tok_labels = []
        for idx, e in enumerate(examples):
            input_ids = e["input_ids"]
            keyphrases_input_ids = e["keyphrases_input_ids"]
            (
                input_ids,
                keyphrases_input_ids,
                keyphrase_indexes,
                replaced_keyphrase_indexes,
                keyphrase_replacement_labels,
                masked_keyphrase_indexes,
                keyphrase_mask_labels,
                keyphrase_mask_num_tok_labels,
            ) = self.replace_keyphrases(input_ids, keyphrases_input_ids)
            # Truncate input ids to max seq len
            input_ids = input_ids[: self.max_seq_len]
            mask_res = self.kp_and_whole_word_mask(
                input_ids,
                keyphrases_input_ids,
                replaced_kp_indexes=replaced_keyphrase_indexes,
            )

            if self.keyphrase_universe_ids is not None:
                # Skip predictions for replacement on Keyphrases that were masked
                mask_skipped_keyphrase_indexes = []
                mask_skipped_replacement_labels = []
                assert len(keyphrase_indexes) == len(keyphrase_replacement_labels)
                for keyphrase_index, label in zip(
                    keyphrase_indexes, keyphrase_replacement_labels
                ):
                    # Skip keyphrases that are out of scope
                    if (
                        not keyphrase_index
                        or len(keyphrase_index) == 0
                        or keyphrase_index[0] >= self.max_seq_len
                    ):
                        continue
                    if (
                        not self.is_keyphrase_masked(mask_res[1], keyphrase_index)
                        and len(mask_skipped_keyphrase_indexes)
                        < self.max_keyphrase_pairs
                    ):
                        # Find boundary word indexes
                        keyphrase_start_idx = (
                            keyphrase_index[0] - 1
                            if (keyphrase_index[0] - 1) >= 0
                            else 0
                        )
                        keyphrase_end_idx = (
                            keyphrase_index[-1] + 1
                            if (keyphrase_index[-1] + 1) < self.max_seq_len
                            else self.max_seq_len - 1
                        )
                        mask_skipped_keyphrase_indexes.append(
                            (keyphrase_start_idx, keyphrase_end_idx)
                        )
                        mask_skipped_replacement_labels.append(label)
                # Truncate to max length
                mask_skipped_keyphrase_indexes = mask_skipped_keyphrase_indexes[
                    : self.max_keyphrase_pairs
                ]
                mask_skipped_replacement_labels = mask_skipped_replacement_labels[
                    : self.max_keyphrase_pairs
                ]
                # Pad pairs to an equal length
                pair_padding_length = self.max_keyphrase_pairs - len(
                    mask_skipped_keyphrase_indexes
                )
                mask_skipped_keyphrase_indexes += [
                    (self.pad_start_index, self.pad_end_index)
                ] * pair_padding_length
                mask_skipped_replacement_labels += [
                    self.label_ignore_index
                ] * pair_padding_length

                # Add to batch
                overall_keyphrase_indexes.append(mask_skipped_keyphrase_indexes)
                overall_keyphrase_replacement_labels.append(
                    mask_skipped_replacement_labels
                )

                if self.do_generation:
                    # Force same length batches
                    catseq_keyphrase_input_ids = e["catseq_keyphrase_input_ids"][
                        : self.max_seq_len
                    ]
                    catseq_keyphrase_decoder_input_ids = e[
                        "catseq_keyphrase_input_ids"
                    ][: self.max_seq_len]
                    catseq_padding_length = self.max_seq_len - len(
                        catseq_keyphrase_input_ids
                    )
                    catseq_keyphrase_input_ids += [
                        self.label_ignore_index
                    ] * catseq_padding_length
                    overall_catseq_keyphrase_input_ids.append(
                        catseq_keyphrase_input_ids
                    )
                    catseq_keyphrase_decoder_input_ids += [
                        self.tokenizer.pad_token_id
                    ] * catseq_padding_length
                    overall_catseq_keyphrase_decoder_input_ids.append(
                        catseq_keyphrase_decoder_input_ids
                    )

            # Truncate to max length
            input_ids = input_ids[: self.max_seq_len]
            pad_mask_label = mask_res[0][: self.max_seq_len]
            pad_kp_mask_label = mask_res[1][: self.max_seq_len]
            # Add padding
            input_padding_length = self.max_seq_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * input_padding_length
            pad_mask_label += [1] * input_padding_length
            pad_kp_mask_label += [1] * input_padding_length

            # Update input ids
            original_input_ids.append(e["input_ids"])
            updated_input_ids.append(input_ids)
            mask_labels.append(pad_mask_label)
            kp_mask_labels.append(pad_kp_mask_label)

            if self.do_keyphrase_infilling:
                # Truncate to max length
                masked_keyphrase_indexes = masked_keyphrase_indexes[
                    : self.max_mask_keyphrase_pairs
                ]
                keyphrase_mask_labels = keyphrase_mask_labels[
                    : self.max_mask_keyphrase_pairs
                ]
                keyphrase_mask_num_tok_labels = keyphrase_mask_num_tok_labels[
                    : self.max_mask_keyphrase_pairs
                ]
                # Add padding if required
                pair_padding_length = self.max_mask_keyphrase_pairs - len(
                    masked_keyphrase_indexes
                )
                masked_keyphrase_indexes += [
                    (self.pad_start_index, self.pad_end_index)
                ] * pair_padding_length
                keyphrase_mask_labels += [
                    ([self.label_ignore_index] * self.kp_max_seq_len)
                    for _ in range(pair_padding_length)
                ]
                keyphrase_mask_num_tok_labels += [
                    self.label_ignore_index
                ] * pair_padding_length

                overall_masked_keyphrase_indexes.append(masked_keyphrase_indexes)
                overall_masked_keyphrase_labels.append(keyphrase_mask_labels)
                overall_keyphrase_mask_num_tok_labels.append(
                    keyphrase_mask_num_tok_labels
                )

        # collate
        # batches with pad token defined in tokenizer
        batch_input = _collate_batch(updated_input_ids, self.tokenizer)
        batch_mask = _collate_batch(mask_labels, self.tokenizer)
        kp_batch_mask = _collate_batch(kp_mask_labels, self.tokenizer)
        # mask
        inputs, labels = self.mask_tokens_and_kp(batch_input, batch_mask, kp_batch_mask)
        if self.keyphrase_universe_ids is not None:
            # batches for keyphrase replacement
            batch_keyphrase_indexes = _collate_batch(
                overall_keyphrase_indexes, self.tokenizer
            )
            batch_keyphrase_replacement_labels = _collate_batch(
                overall_keyphrase_replacement_labels, self.tokenizer
            )

        if self.do_keyphrase_infilling:
            batch_masked_keyphrase_indexes = _collate_batch(
                overall_masked_keyphrase_indexes, self.tokenizer
            )
            batch_masked_keyphrase_labels = self._collate_label_batch(
                overall_masked_keyphrase_labels, self.tokenizer
            )
            batch_keyphrase_mask_num_tok_labels = _collate_batch(
                overall_keyphrase_mask_num_tok_labels, self.tokenizer
            )

        if self.do_generation:
            batch_catseq_keyphrase_decoder_inputs = _collate_batch(
                overall_catseq_keyphrase_decoder_input_ids, self.tokenizer
            )
            batch_catseq_keyphrase_labels = self._collate_label_batch(
                overall_catseq_keyphrase_input_ids, self.tokenizer
            )
            batch_original_labels = self._collate_label_batch(
                original_input_ids, self.tokenizer
            )
            if self.use_bart:
                if self.do_keyphrase_generation:
                    return {
                        "input_ids": inputs,
                        "labels": batch_catseq_keyphrase_labels,
                    }
                else:
                    return {"input_ids": inputs, "labels": batch_original_labels}

            return {
                "input_ids": inputs,
                "decoder_input_ids": batch_catseq_keyphrase_decoder_inputs,
                "labels": batch_catseq_keyphrase_labels,
                "keyphrase_pairs": batch_keyphrase_indexes,
                "replacement_labels": batch_keyphrase_replacement_labels,
            }
        if self.do_keyphrase_infilling:
            if self.keyphrase_universe_ids:
                return {
                    "input_ids": inputs,
                    "labels": labels,
                    "keyphrase_pairs": batch_keyphrase_indexes,
                    "replacement_labels": batch_keyphrase_replacement_labels,
                    "masked_keyphrase_pairs": batch_masked_keyphrase_indexes,
                    "masked_keyphrase_labels": batch_masked_keyphrase_labels,
                    "keyphrase_mask_num_tok_labels": batch_keyphrase_mask_num_tok_labels,
                }
            else:
                return {
                    "input_ids": inputs,
                    "labels": labels,
                    "masked_keyphrase_pairs": batch_masked_keyphrase_indexes,
                    "masked_keyphrase_labels": batch_masked_keyphrase_labels,
                    "keyphrase_mask_num_tok_labels": batch_keyphrase_mask_num_tok_labels,
                }
        if self.keyphrase_universe_ids:
            return {
                "input_ids": inputs,
                "labels": labels,
                "keyphrase_pairs": batch_keyphrase_indexes,
                "replacement_labels": batch_keyphrase_replacement_labels,
            }
        else:
            return {"input_ids": inputs, "labels": labels}

    def _collate_label_batch(self, examples, tokenizer):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        # Check if padding is necessary.
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        result = examples[0].new_full(
            [len(examples), max_length], self.label_ignore_index
        )
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result

    @staticmethod
    def is_keyphrase_masked(keyphrase_masks, keyphrase_indexes):
        for keyphrase_index in keyphrase_indexes:
            if (
                keyphrase_index < len(keyphrase_masks)
                and keyphrase_masks[keyphrase_index] == 0
            ):
                return False
        return True

    def replace_keyphrases(self, input_ids, keyphrases_input_ids):
        """
        Replace a defined percentage of keyphrases with another keyphrase
        from the keyphrase universe
        """
        if (
            self.keyphrase_universe_ids is None or self.kp_replace_percentage == 0
        ) and self.do_keyphrase_infilling == False:
            return input_ids, keyphrases_input_ids, [], [], [], [], [], []
        new_input_ids = []
        new_keyphrase_input_ids = []
        keyphrase_replacement_labels = []
        replaced_keyphrase_indexes = []
        keyphrase_indexes = []
        masked_keyphrase_indexes = []
        keyphrase_mask_labels = []
        keyphrase_mask_num_tok_labels = []
        # Replace Keyphrases in Input
        input_idx = 0
        while input_idx < len(input_ids):
            input_id = input_ids[input_idx]
            found_keyphrase = False
            for keyphrase in keyphrases_input_ids:
                keyphrase_idx = input_idx + len(keyphrase)
                if input_ids[input_idx:keyphrase_idx] == keyphrase:
                    found_keyphrase = True
                    curr_new_input_idx = len(new_input_ids)
                    if (
                        self.do_keyphrase_infilling
                        and random.random() < self.kp_infill_percentage
                    ):
                        # Choose a random keyphrase from the keyphrase universe
                        replaced_keyphrase = [self.tokenizer.mask_token_id]
                        # Compute Indexes
                        start_idx = (
                            curr_new_input_idx - 1
                            if (curr_new_input_idx - 1) >= 0
                            else 0
                        )
                        if start_idx >= self.max_seq_len:
                            continue
                        end_idx = (
                            curr_new_input_idx + 1
                            if (curr_new_input_idx + 1) < self.max_seq_len
                            else self.max_seq_len - 1
                        )
                        masked_keyphrase_indexes.append([start_idx, end_idx])
                        # Update Input IDs with replaced keyphrase
                        new_input_ids.extend(replaced_keyphrase)
                        # Capture the number of tokens label
                        kp_num_tok = (
                            len(keyphrase)
                            if len(keyphrase) < self.kp_max_seq_len
                            else self.kp_max_seq_len - 1
                        )
                        keyphrase_mask_num_tok_labels.append(kp_num_tok)
                        # Update Keyphrase Input IDs
                        kp_pad_len = self.kp_max_seq_len - len(keyphrase)
                        kp_label = keyphrase
                        if kp_pad_len < 0:
                            kp_label = keyphrase[: self.kp_max_seq_len]
                        else:
                            kp_label += [self.label_ignore_index] * kp_pad_len
                        keyphrase_mask_labels.append(kp_label)
                    # Consolidate input IDs uptil now
                    elif (
                        self.keyphrase_universe_ids is not None
                        and random.random() < self.kp_replace_percentage
                    ):
                        # Choose a random keyphrase from the keyphrase universe
                        replaced_keyphrase = random.choice(self.keyphrase_universe_ids)
                        # Make sure we aren't replacing with the same keyphrase
                        while replaced_keyphrase == keyphrase:
                            replaced_keyphrase = random.choice(
                                self.keyphrase_universe_ids
                            )
                        # Compute Indexes
                        indexes = [
                            idx
                            for idx in range(
                                curr_new_input_idx,
                                curr_new_input_idx + len(replaced_keyphrase),
                            )
                        ]
                        keyphrase_indexes.append(indexes)
                        replaced_keyphrase_indexes.append(indexes)
                        # Update Input IDs with replaced keyphrase
                        new_input_ids.extend(replaced_keyphrase)
                        # Update Keyphrase Input IDs
                        new_keyphrase_input_ids.append(replaced_keyphrase)
                        # Capture the replacement label
                        keyphrase_replacement_labels.append(1)
                    else:
                        # Compute Indexes
                        indexes = [
                            idx
                            for idx in range(
                                curr_new_input_idx, curr_new_input_idx + len(keyphrase)
                            )
                        ]
                        keyphrase_indexes.append(indexes)
                        # Update Input IDs with original keyphrase
                        new_input_ids.extend(keyphrase)
                        # Update Keyphrase Input IDs
                        new_keyphrase_input_ids.append(keyphrase)
                        # Capture the non-replacement label
                        keyphrase_replacement_labels.append(0)
                    # Skip Input to after Keyphrase
                    input_idx = keyphrase_idx
                    break
            if not found_keyphrase:
                new_input_ids.append(input_id)
                input_idx += 1

        return (
            new_input_ids,
            new_keyphrase_input_ids,
            keyphrase_indexes,
            replaced_keyphrase_indexes,
            keyphrase_replacement_labels,
            masked_keyphrase_indexes,
            keyphrase_mask_labels,
            keyphrase_mask_num_tok_labels,
        )

    def kp_and_whole_word_mask(
        self,
        input_tokens,
        kp_tokens_list,
        max_predictions=512,
        replaced_kp_indexes=None,
    ):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = []
        kp_indexes = []
        for (i, token) in enumerate(input_tokens):
            if (
                token == self.tokenizer.cls_token_id
                or token == self.tokenizer.sep_token_id
                or token == self.tokenizer.mask_token_id
            ):
                continue
            kp_flag = False
            for kp in kp_tokens_list:  # kp = ["KP1-T1", "KP1-T2"]
                j = i + len(kp)
                if j < len(input_tokens):
                    if input_tokens[i:j] == kp:  # input_tokens = ["KP1-T1", "KP1-T2"]
                        kp_indexes.append(
                            [x for x in range(i, j)]
                        )  # kp_indexes = ["index of KP1-T1", "index of KP1-T2"]
                        i = j - 1
                        kp_flag = True
                        break
            if (
                kp_flag
            ):  # if token is included in kp mask then don't include in random token mask
                continue
            if self.tokenizer._convert_id_to_token(token).startswith("Ġ"):
                cand_indexes.append([i])
            else:
                if len(cand_indexes) >= 1:
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])

        if replaced_kp_indexes:
            filtered_kp_indexes = [
                kp_index
                for kp_index in kp_indexes
                if kp_index not in replaced_kp_indexes
            ]
        else:
            filtered_kp_indexes = kp_indexes

        tok_to_predict = min(
            max_predictions,
            max(
                1,
                int(
                    round(
                        len(input_tokens)
                        * (1 - self.kp_mask_percentage)
                        * self.mlm_probability
                    )
                ),
            ),
        )
        # Probability of masking keyphrases is KP_MASK_PERCENTAGE * MLM_PROBABILITY over the total number
        # of tokens in the document (input_tokens) to make sure we are masking ~15% of tokens in line
        # with all other language modelling pre-training literature
        kp_to_predict = min(
            max_predictions,
            max(
                1,
                int(
                    round(
                        len(input_tokens)
                        * self.kp_mask_percentage
                        * self.mlm_probability
                    )
                ),
            ),
        )

        tok_mask_labels = self.get_mask_labels(
            cand_indexes=cand_indexes,
            len_input_tokens=len(input_tokens),
            num_to_predict=tok_to_predict,
        )
        kp_mask_labels = self.get_mask_labels(
            cand_indexes=filtered_kp_indexes,
            len_input_tokens=len(input_tokens),
            num_to_predict=kp_to_predict,
        )
        return tok_mask_labels, kp_mask_labels

    def get_mask_labels(self, cand_indexes, len_input_tokens, num_to_predict):
        random.shuffle(cand_indexes)
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [
            1 if i in covered_indexes else 0 for i in range(len_input_tokens)
        ]
        return mask_labels

    def mask_tokens_and_kp(self, inputs, mask_labels, kp_mask_labels):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language "
                "modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training
        # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels
        kp_probability_matrix = kp_mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        # do zero for special tokens
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        kp_probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )

        # assert kp_probability_matrix & probability_matrix == 0
        # do zero for padded points
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
            kp_probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        kp_masked_indices = kp_probability_matrix.bool()
        # get the gold lables
        labels[
            ~(masked_indices | kp_masked_indices)
        ] = (
            self.label_ignore_index
        )  # We only compute loss on random masked tokens and kp masked token else is set to -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )
        # 80 % masking for keyphrases
        kp_indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & kp_masked_indices
        )
        inputs[kp_indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )
        # generate random tokens
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        inputs[indices_random] = random_words[indices_random]

        # replace 10 # kp tokens with random indices
        # TODO: If keyphrase_universe available, replace with another keyphrase and capture
        # indices to classify replacement
        kp_indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & kp_masked_indices
            & ~kp_indices_replaced
        )
        inputs[kp_indices_random] = random_words[kp_indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        # print("inside mask tok functiom \n",inputs,"\n", labels,"\n")

        # generation - t1, t2, t3 (actual) - [MASK], t4 [MASK], t5, t6
        # replacement - t1, t2, t3 (actual) - [MASK], t4 [MASK], t5, t6 (replace) t9

        return inputs, labels
