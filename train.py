import logging
import os
import sys

from datasets import load_dataset, load_from_disk, concatenate_datasets, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from transformers import DataCollatorWithPadding, EvalPrediction
from transformers import HfArgumentParser, TrainingArguments
from arguments import ModelArguments, DataTrainingArguments

from utils import set_seed
from utils_qa import tokenize_into_morphs, check_no_error
from utils_qa import get_column_names, preprocess_features_of_Dataset, prepare_train_features, prepare_validation_features
from utils_qa import post_processing_function, postprocess_qa_predictions
from trainer_qa import QuestionAnsweringTrainer
from retrieval import BM25SparseRetriever

logger = logging.getLogger(__name__)


def main():
    # Arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # Set seed first.
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Training/evaluation parameters %s", training_args)



    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    model.to("cuda")
    print("model uses device:", model.device)

    # Load dataset from disk
    datasets = load_from_disk(data_args.dataset_name)
    last_checkpoint, data_args.max_seq_length = check_no_error(training_args, data_args, tokenizer, datasets)
    print(datasets)
    # korquad dataset
    korquad_datasets = load_dataset("squad_kor_v1")
    
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation"]
    column_names = train_dataset.column_names
    # korquad dataset
    korquad_train_dataset = korquad_datasets["train"]
    korquad_eval_dataset = korquad_datasets["validation"]
    column_names = get_column_names()

    # Run passage retrieval if do evaluation
    if training_args.do_eval:
        retriever = BM25SparseRetriever()
        eval_dataset = retriever.get_standard_dataset_for_QA(eval_dataset, topk=data_args.topk)["validation"] # id, question, context
        eval_dataset = eval_dataset.map(
            lambda example: prepare_validation_features(example, tokenizer, data_args), # tokenize dataset
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    # else, use all datasets for training
    else:
        train_dataset = concatenate_datasets([
            preprocess_features_of_Dataset(train_dataset).flatten_indices(),
            preprocess_features_of_Dataset(eval_dataset).flatten_indices(),
            preprocess_features_of_Dataset(korquad_train_dataset).flatten_indices(), # korquad dataset
            preprocess_features_of_Dataset(korquad_eval_dataset).flatten_indices(), # korquad dataset
        ])

    train_dataset = train_dataset.map(
        lambda example: prepare_train_features(example, tokenizer, data_args), # tokenize dataset
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # if not padded to max length, do pad with data collator.
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Metric
    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,         # tokenized question-context sequence
        eval_dataset=eval_dataset if training_args.do_eval else None,            # tokenized question-context sequence
        eval_examples=datasets["validation"] if training_args.do_eval else None, # before tokenization; including answers
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
