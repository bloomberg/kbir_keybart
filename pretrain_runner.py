import argparse
import math
import os
import sys
import torch
import logging
import time
import traceback
from datasets import load_dataset
from glob import glob
import transformers
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
from transformers import RobertaTokenizer, RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import set_seed
from transformers import PrinterCallback
from transformers import DataCollatorForWholeWordMask

from transformers import BartTokenizer, BartTokenizerFast
from transformers import BartForConditionalGeneration

from utils.data_collators import DataCollatorForKLM
from model.model import KLMForReplacementAndMaskedLM, EncoderDecoderModel
from trainer.trainer import TrainerWithEvalCollator

logger = logging.getLogger(__name__)


def is_main_process(local_rank):
    """
    Whether or not the current process is the local process, based on `local_rank`.
    """
    return local_rank in [-1, 0]


def main(args):

    if args.do_eval:
        assert args.eval_data_dir is not None

    training_args = TrainingArguments(
        no_cuda=args.no_cuda,
        output_dir=args.model_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        local_rank=args.local_rank,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        do_train=args.do_train,
        logging_steps=args.logging_steps,
        disable_tqdm=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        + f"16-bits optimization level {training_args.fp16_opt_level}"
    )
    logger.warning(f"Process rank from env: {os.getenv('RANK')}")
    logger.info("There are %d GPU(s) available.", torch.cuda.device_count())
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Choose this when trying to train from scratch
    # config
    # config = RobertaConfig.from_pretrained(args.roberta_mlm_model_dir)
    # model roberta
    # model = RobertaForMaskedLM(config=config)

    # Choose this when starting from an already pretrained model
    if (
        args.do_keyphrase_replacement or args.do_keyphrase_infilling
    ) and not args.do_generation:
        logger.info("Loading KLMForReplacementAndMaskedLM")
        model = KLMForReplacementAndMaskedLM.from_pretrained(
            args.roberta_mlm_model_dir,
            use_doc_emb=args.use_doc_emb,
            kp_max_seq_len=args.kp_max_seq_len,
            mlm_loss_weight=args.mlm_loss_weight,
            replacement_loss_weight=args.replacement_loss_weight,
            keyphrase_infill_loss_weight=args.keyphrase_infill_loss_weight,
            infill_num_tok_loss_weight=args.infill_num_tok_loss_weight,
        )
    elif args.do_generation:
        if args.use_bart:
            logger.info("Loading BART")
            model = BartForConditionalGeneration.from_pretrained(args.bart_model_dir)
        else:
            raise NotImplementedError(
                "Generation models outside of BART are currently not supported!"
            )
    else:
        logger.info("Loading RobertaForMaskedLM")
        model = RobertaForMaskedLM.from_pretrained(args.roberta_mlm_model_dir)

    logger.info("Loaded pre-trained model")

    freeze_layers = list(range(args.num_frozen_layers))
    if freeze_layers:
        for name, param in model.roberta.encoder.layer.named_children():
            if int(name) in freeze_layers:
                logger.info("Freezing Layer: ", name)
                param.requires_grad = False

    if args.do_generation and args.use_bart:
        tokenizer = BartTokenizer.from_pretrained(
            args.bart_model_dir, use_fast=True, max_length=512
        )
    # this tokenizer needs to be downloaded and we need to point to the path
    else:
        tokenizer = RobertaTokenizer.from_pretrained(
            args.roberta_tokenizer_dir, use_fast=True, max_length=512
        )

    keyphrase_universe = set()
    keyphrase_universe_ids = None
    if args.do_keyphrase_replacement:
        logger.info("Loading Keyphrase Universe")
        with open(args.keyphrase_universe) as f:
            for line in f:
                keyphrase_universe.add(line)
                # Restrict size of keyphrase universe for computational reasons
                if (
                    args.keyphrase_universe_size != -1
                    and len(keyphrase_universe) == args.keyphrase_universe_size
                ):
                    break
        keyphrase_universe = list(keyphrase_universe)
        keyphrase_universe_ids = tokenizer(
            keyphrase_universe, truncation=True, add_special_tokens=False,
        )["input_ids"]
        assert len(keyphrase_universe) == len(keyphrase_universe_ids)

    def parse_keyphrases(text, keywords):
        keyphrases = []
        for keyphrase in keywords.split(" , "):
            if not keyphrase.strip():
                continue
            keyphrase_index = text.find(keyphrase)
            if keyphrase_index == -1:
                keyphrase_index = text.lower().find(keyphrase)
                # Can't find keyphrase in text
                if keyphrase_index == -1:
                    continue
                keyphrase = text[keyphrase_index : keyphrase_index + len(keyphrase)]
            # Decide whether a space is required before for the tokenizer to have a consistent behavior
            if keyphrase_index > 0:
                if text[keyphrase_index - 1] == " ":
                    keyphrase = " " + keyphrase
            keyphrases.append(keyphrase)

        return keyphrases

    def get_catseq_keyphrases(keywords):
        keyphrases = ";".join(list(set(keywords.split(" , "))))

        return keyphrases

    def tokenize_klm_function(examples):
        try:
            text_data = [
                title + ". " + abstract
                for title, abstract in zip(examples["title"], examples["abstract"])
            ]
            examples_batch_encoding = tokenizer(
                text_data,
                truncation=True,
                return_special_tokens_mask=True,
                max_length=512,
            )
            keyphrases_input_ids = []
            catseq_keyphrases_input_ids = []
            for text, keyphrase_list in zip(text_data, examples["keywords"]):
                keyphrases = parse_keyphrases(text, keyphrase_list)
                keyphrase_input_ids = tokenizer(
                    keyphrases, truncation=True, add_special_tokens=False,
                )["input_ids"]
                keyphrases_input_ids.append(keyphrase_input_ids)
                catseq_keyphrases = get_catseq_keyphrases(keyphrase_list)
                catseq_keyphrase_input_ids = tokenizer(
                    catseq_keyphrases, truncation=True, add_special_tokens=False,
                )["input_ids"]
                catseq_keyphrases_input_ids.append(catseq_keyphrase_input_ids)
            examples_batch_encoding["keyphrases_input_ids"] = keyphrases_input_ids
            examples_batch_encoding[
                "catseq_keyphrase_input_ids"
            ] = catseq_keyphrases_input_ids
            return examples_batch_encoding
        except Exception as e:
            logger.info("Skipping batch due to errors")
            logger.info(e)

    def tokenize_mlm_function(examples):
        text = [
            title + ". " + abstract
            for title, abstract in zip(examples["title"], examples["abstract"])
        ]
        return tokenizer(text, truncation=True, max_length=512,)

    def filter_empty_keyphrases(example):
        text = example["title"] + ". " + example["abstract"]
        try:
            if (
                not example["keywords"]
                or not example["title"]
                or not example["abstract"]
                or not parse_keyphrases(text, example["keywords"])
            ):
                return False
            return True
        except:
            return False

    if training_args.do_eval:
        logger.info("Initializing Eval Dataset")
        eval_dataset = load_dataset(
            "json", data_files=glob(args.eval_data_dir + "/*.txt")
        )

        logger.info("Initializing Eval DataCollator")
        if args.eval_task == "KLM":
            logger.info("Filter Eval Dataset")
            eval_dataset = eval_dataset.filter(filter_empty_keyphrases,)

            logger.info("Tokenize Eval Dataset")
            tokenized_eval_dataset = eval_dataset.map(
                tokenize_klm_function,
                batched=True,
                remove_columns=["title", "abstract", "keywords"],
                writer_batch_size=3_000,
                load_from_cache_file=False,
            )
            logger.info("Setting up Eval DataCollator")
            eval_data_collator = DataCollatorForKLM(
                tokenizer=tokenizer,
                keyphrase_universe_ids=keyphrase_universe_ids,
                mlm_probability=args.mlm_probability,
                kp_mask_percentage=args.keyphrase_mask_percentage,
                kp_replace_percentage=args.keyphrase_replace_percentage,
                max_keyphrase_pairs=args.max_keyphrase_pairs,
                max_seq_len=args.max_seq_len,
                do_generation=args.do_generation,
                use_bart=args.use_bart,
                do_keyphrase_generation=args.do_keyphrase_generation,
                do_keyphrase_infilling=args.do_keyphrase_infilling,
                kp_max_seq_len=args.kp_max_seq_len,
                max_mask_keyphrase_pairs=args.max_mask_keyphrase_pairs,
            )

        elif args.eval_task == "MLM":
            logger.info("Tokenize Eval Dataset")
            tokenized_eval_dataset = eval_dataset.map(
                tokenize_mlm_function,
                batched=True,
                remove_columns=["title", "abstract"],
                writer_batch_size=3_000,
                load_from_cache_file=False,
            )
            logger.info("Setting up Eval DataCollator")
            eval_data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer)

    logger.info(f"Task = {args.task}")
    if training_args.do_train:
        logger.info("Initializing Train Dataset")
        train_dataset = load_dataset(
            "json", data_files=glob(args.train_data_dir + "/*.txt")
        )

        logger.info("Initializing Train DataCollator")
        if args.task == "KLM":
            logger.info("Filter Train Dataset")
            train_dataset = train_dataset.filter(filter_empty_keyphrases,)
            logger.info(train_dataset)
            logger.info("Tokenize Train Dataset")
            tokenized_train_dataset = train_dataset.map(
                tokenize_klm_function,
                batched=True,
                remove_columns=["title", "abstract", "keywords"],
                writer_batch_size=3_000,
                load_from_cache_file=False,
            )
            logger.info(tokenized_train_dataset)
            logger.info("Setting up Train DataCollator")
            train_data_collator = DataCollatorForKLM(
                tokenizer=tokenizer,
                keyphrase_universe_ids=keyphrase_universe_ids,
                mlm_probability=args.mlm_probability,
                kp_mask_percentage=args.keyphrase_mask_percentage,
                kp_replace_percentage=args.keyphrase_replace_percentage,
                max_keyphrase_pairs=args.max_keyphrase_pairs,
                max_seq_len=args.max_seq_len,
                do_generation=args.do_generation,
                use_bart=args.use_bart,
                do_keyphrase_generation=args.do_keyphrase_generation,
                do_keyphrase_infilling=args.do_keyphrase_infilling,
                kp_max_seq_len=args.kp_max_seq_len,
                max_mask_keyphrase_pairs=args.max_mask_keyphrase_pairs,
            )

        elif args.task == "MLM":
            logger.info("Tokenize Train Dataset")
            tokenized_train_dataset = train_dataset.map(
                tokenize_mlm_function,
                batched=True,
                remove_columns=["title", "abstract"],
                writer_batch_size=3_000,
                load_from_cache_file=False,
            )
            logger.info("Setting up Train DataCollator")
            train_data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer)

    logger.info("Initializing Trainer")
    if args.do_eval:
        trainer = TrainerWithEvalCollator(
            model=model,
            args=training_args,
            data_collator=train_data_collator if training_args.do_train else None,
            eval_data_collator=eval_data_collator if training_args.do_eval else None,
            train_dataset=tokenized_train_dataset if training_args.do_train else None,
            eval_dataset=tokenized_eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
        )                   
    else:               
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=train_data_collator if training_args.do_train else None,
            train_dataset=tokenized_train_dataset if training_args.do_train else None,
            tokenizer=tokenizer,
        )
    trainer.add_callback(PrinterCallbackWithFlush)

    if training_args.do_train:
        logger.info("Training Model")
        checkpoint = None
        if args.is_checkpoint:
            logger.info("Loading from checkpoint")
            checkpoint = (
                args.roberta_mlm_model_dir
                if args.roberta_mlm_model_dir
                else args.bart_model_dir
            )
        trainer.train(resume_from_checkpoint=checkpoint)

    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results_mlm_wwm.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")


class PrinterCallbackWithFlush(PrinterCallback):
    def __init__(self):
        self.prev_steps = None
        self.prev_time = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        curr_steps = state.global_step
        curr_time = time.time()
        if self.prev_steps:
            print(
                f"Steps since last log: {curr_steps - self.prev_steps}, "
                f"Global steps: {curr_steps}, "
                f"Max steps: {state.max_steps}, "
                f"Time since last log: {curr_time - self.prev_time}",
                flush=True,
            )
        else:
            print("Starting steps and time history", flush=True)
        self.prev_steps = curr_steps
        self.prev_time = curr_time

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        perplexity = math.exp(metrics["eval_loss"])
        print({"perplexity": perplexity})


def world_size():
    """Returns the total number of processes in a distributed job (num_nodes x gpus_per_node).
    Returns 1 in a non-distributed job.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_distributed():
    """Returns True iff this is a distributed job (more than one process)."""
    return world_size() > 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data-dir",
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--eval-data-dir",
        type=str,
        required=False,
        help="The eval data dir. Required if --do-eval is set.",
    )
    parser.add_argument(
        "--keyphrase-universe",
        type=str,
        required=False,
        help="File containing all the keyphrases in across the train and dev set used for keyphrase replacement.",
    )
    parser.add_argument(
        "--model-dir",
        default="./",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--roberta-tokenizer-dir",
        type=str,
        required=False,
        default=None,
        help="The directory from where the RoBERTa tokenizer is loaded.",
    )
    parser.add_argument(
        "--roberta-mlm-model-dir",
        type=str,
        required=False,
        default=None,
        help="The directory from where the pre-trained RoBERTa model is loaded.",
    )
    parser.add_argument(
        "--bart-model-dir",
        type=str,
        required=False,
        default=None,
        help="The directory from where the pre-trained RoBERTa model is loaded.",
    )
    parser.add_argument(
        "--epochs", default=3.0, type=float, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--learning-rate",
        default=5e-5,
        type=float,
        help="Learning rate to use for training the model",
    )
    parser.add_argument(
        "--num-frozen-layers",
        default=0,
        type=int,
        help="Number of RoBERTa encoder layers to freeze during training",
    )
    parser.add_argument(
        "--adam-epsilon",
        default=1e-8,
        type=float,
        help="The epsilon hyperparameter for the Adam optimizer",
    )
    parser.add_argument(
        "--train-batch-size",
        default=64,
        type=int,
        help="Training dataset batch size per GPU",
    )
    parser.add_argument(
        "--eval-batch-size",
        default=128,
        type=int,
        help="Eval dataset batch size per GPU",
    )
    parser.add_argument(
        "--warmup-steps",
        default=0,
        type=int,
        help="Number of steps used for a linear warmup from 0 to learning_rate.",
    )
    parser.add_argument(
        "--max-steps",
        default=-1,
        type=int,
        help="If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save-steps",
        default=500,
        type=int,
        help="Number of updates steps before two checkpoint saves.",
    )
    parser.add_argument(
        "--logging-steps",
        default=500,
        type=int,
        help="Number of updates steps before logs",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do-train", action="store_true", help="Whether to run training or not."
    )
    parser.add_argument(
        "--do-eval", action="store_true", help="Whether to run evaluation or not."
    )
    parser.add_argument(
        "--is-checkpoint",
        action="store_true",
        help="Whether to treat the model path as a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Run evaluation every these many steps",
    )
    parser.add_argument(
        "--dataloader-num-workers", type=int, default=0, help="dataloader num workers"
    )

    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank when doing distributed training, set to -1 if running non-distributed",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=int(os.getenv("RANK", 0)),
        help="Rank when doing distributed training, doesn't matter for non-distributed",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=world_size(),
        help="world size when doing distributed training, set to -1 if running non-distributed",
    )
    parser.add_argument(
        "--task",
        choices=["KLM", "MLM"],
        default="KLM",
        help="KLM training or whole word masking training",
    )
    parser.add_argument(
        "--eval-task",
        choices=["KLM", "MLM"],
        default="MLM",
        help="What masking to use for eval: KLM/MLM",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
    )

    parser.add_argument(
        "--do-keyphrase-replacement",
        action="store_true",
        help="Whether to enable keyphrase replacement during KLM.",
    )
    parser.add_argument(
        "--do-keyphrase-generation",
        action="store_true",
        help="Whether to have the generation labels correspond to CatSeq representation of keyphrases.",
    )
    parser.add_argument(
        "--do-keyphrase-infilling",
        action="store_true",
        help="Whether to use the text in-filling pre-training setup for BART.",
    )
    parser.add_argument(
        "--keyphrase-universe-size",
        type=int,
        default=-1,
        help="Size of universe used during keyphrase replacement during KLM.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max Sequence Length considered for an input",
    )
    parser.add_argument(
        "--kp-max-seq-len",
        type=int,
        default=10,
        help="Max Sequence Length considered for an keyphrase during infilling",
    )
    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="Probability of masking a token in the input during pre-training",
    )
    parser.add_argument(
        "--keyphrase-mask-percentage",
        type=float,
        default=0.4,
        help="If training on the KLM objective, percentage of keyphrases tokens to mask as a percentage of a total input tokens. When used with infilling this is the percentage of keyphrases to mask.",
    )
    parser.add_argument(
        "--keyphrase-replace-percentage",
        type=float,
        default=0.1,
        help="If training on the KLM objective and replacing keyphrases, percentage of keyphrases to replace as a percentage of a total keyphrases",
    )
    parser.add_argument(
        "--max-keyphrase-pairs",
        type=int,
        default=20,
        help="If training on the KLM objective and replacing keyphrases, max number of keyphrases to consider in replacement task",
    )
    parser.add_argument(
        "--max-mask-keyphrase-pairs",
        type=int,
        default=10,
        help="If training on the KLM objective and infilling keyphrases, max number of keyphrases to consider for masking",
    )
    parser.add_argument(
        "--use-doc-emb",
        action="store_true",
        help="Whether to use [CLS] document embedding during keyphrase replacement.",
    )
    parser.add_argument(
        "--do-generation",
        action="store_true",
        help="Whether to set up pre-training as an denoisining autoencoder with a decoder head.",
    )
    parser.add_argument(
        "--use-bart",
        action="store_true",
        help="Whether to use bart when --do-generation is set, needs --bart-model-dir to be non-null",
    )
    parser.add_argument(
        "--mlm-loss-weight",
        type=float,
        default=1.0,
        help="Co-efficient for masked language modelling loss in overall loss",
    )
    parser.add_argument(
        "--replacement-loss-weight",
        type=float,
        default=1.0,
        help="Co-efficient for keyphrase replacement classification loss in overall loss",
    )
    parser.add_argument(
        "--keyphrase-infill-loss-weight",
        type=float,
        default=1.0,
        help="Co-efficient for keyphrase infilling loss in overall loss",
    )
    parser.add_argument(
        "--infill-num-tok-loss-weight",
        type=float,
        default=1.0,
        help="Co-efficient for keyphrase number of token classification loss in overall loss",
    )
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--fp16_opt_level", type=str, default="O2")

    args = parser.parse_args()

    if is_distributed():
        print("Process is distributed")
        # args.local_rank = int(os.getenv('LOCAL_RANK'))
        # To avoid deadlocks on the tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"args.local_rank = {args.local_rank}")
    print(f"args.rank = {args.rank}")
    print(f"args.world_size = {args.world_size}")
    print(f"os.getenv('RANK') = {os.getenv('RANK')}")
    print(f"os.getenv('LOCAL_RANK') = {os.getenv('LOCAL_RANK')}")
    print(f"os.getenv('WORLD_SIZE') = {os.getenv('WORLD_SIZE')}")
    print(args)

    for _ in range(5):
        try:
            main(args)
            break
        except Exception as e:
            print(e)
            traceback.print_exc()
            print("Trying to reconnect to master")
            time.sleep(1)
    else:
        raise RuntimeError("Could not successfully finish job")
