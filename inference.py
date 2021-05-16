import logging
import os
import sys

from datasets import load_from_disk, load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from transformers import DataCollatorWithPadding, EvalPrediction
from transformers import HfArgumentParser, TrainingArguments
from arguments import ModelArguments, DataTrainingArguments

from utils import set_seed
from utils_qa import check_no_error, prepare_validation_features
from utils_qa import post_processing_function, postprocess_qa_predictions
from trainer_qa import QuestionAnsweringTrainer
from retrieval import BM25SparseRetriever

logger = logging.getLogger(__name__)


def main():
    # Arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"data is from {data_args.dataset_name}")
    print(f"model is from {model_args.model_name_or_path}")

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
    examples = load_from_disk(data_args.dataset_name)["validation"] # id, question
    print(examples)

    # Run passage retrieval, make dataset for model input
    retriever = BM25SparseRetriever()
    datasets = retriever.get_standard_dataset_for_QA(examples, topk=data_args.topk) # validation: id, question, context

    # Check dataset
    last_checkpoint, data_args.max_seq_length = check_no_error(training_args, data_args, tokenizer, datasets)

    # Pre-process for dataset
    column_names = datasets["validation"].column_names
    eval_dataset = datasets["validation"]
    eval_dataset = eval_dataset.map(
        lambda example: prepare_validation_features(example, tokenizer, data_args), # tokenize dataset
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
    print("Init Trainer...")
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset= None,
        eval_dataset=eval_dataset,            # tokenized question-context sequence
        eval_examples=datasets["validation"], # before tokenization; id, question, context
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function, # post-processing
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")
    #### eval dataset & eval example - will create predictions.json
    predictions = trainer.predict(
        test_dataset=eval_dataset,
        test_examples=datasets["validation"],
        max_answer_length=data_args.max_answer_length,
    )

    # predictions.json is already saved when we call postprocess_qa_predictions(). so there is no need to further use predictions.
    print("No metric can be presented because there is no correct answer given. Job done!")

if __name__ == "__main__":
    main()
