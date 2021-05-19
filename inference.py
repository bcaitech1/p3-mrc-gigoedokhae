import logging
import os
import sys

from datasets import load_from_disk, load_metric
from transformers import DataCollatorWithPadding, EvalPrediction

from utils import set_seed, get_MDT_parsers, get_CTM
from utils_qa import check_no_error, prepare_validation_features
from utils_qa import post_processing_function, postprocess_qa_predictions
from trainer_qa import QuestionAnsweringTrainer
from retrieval import BM25SparseRetriever

logger = logging.getLogger(__name__)


def initialize(data_args, training_args):
    # Set seed first.
    # Do 'import random' and 'print random.random()' after set_seed() called,
    # then you can see that the same number is generated every time initialize() called.
    set_seed(training_args.seed) # default = 42

    # Set logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Initialize data and inference(training) arguments
    data_args.dataset_path = "/opt/ml/input/data/test_dataset"
    intialize_inference_args(training_args) # variable name is training_args, not inference_args.
    
def intialize_inference_args(training_args):
    training_args.do_predict = True
    training_args.do_train = False
    training_args.do_eval = False
    training_args.evaluation_strategy = "no"


def main():
    # SETTINGS
    # arguments and initialization
    model_args, data_args, training_args = get_MDT_parsers()
    initialize(data_args, training_args)
    print(f"data is from {data_args.dataset_path}")
    print(f"model is from {model_args.model_path}")

    # config, tokenizer and model
    config, tokenizer, model = get_CTM(model_args)
    model.eval()
    print("model uses device:", model.device)

    # DATA
    # examples: id, question
    # type: Dataset <- DatasetDict["validation"]
    examples = load_from_disk(data_args.dataset_path)["validation"]
    print(examples)

    # retrieved dataset: id, question, top-k contexts
    # type: DatasetDict with key = ["validation"] (standard dataset format for QA)
    retriever = BM25SparseRetriever()
    retrieved_dataset = retriever.retrieve_standard_dataset_for_QA(examples, topk=training_args.topk)

    # check dataset
    last_checkpoint, data_args.max_seq_length = check_no_error(training_args, data_args, tokenizer, retrieved_dataset)

    # eval dataset: tokenized (question + context) sequences <- prepare_validation_features
    column_names = retrieved_dataset["validation"].column_names
    eval_dataset = retrieved_dataset["validation"]
    eval_dataset = eval_dataset.map(
        lambda example: prepare_validation_features(example, tokenizer, data_args),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # ETC
    # data collator
    # if not padded to max length, do pad with data collator.
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # metric
    metric = load_metric("squad")
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # INFERENCE
    print("Init Trainer...")
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset= None,
        eval_dataset=eval_dataset,
        eval_examples=retrieved_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # PREDICTION
    logger.info("*** Evaluate ***")
    #### eval dataset & eval example - will create predictions.json
    predictions = trainer.predict(
        test_dataset=eval_dataset,
        test_examples=retrieved_dataset["validation"],
        max_answer_length=data_args.max_answer_length,
    )

    # predictions.json is already saved when we call postprocess_qa_predictions(). so there is no need to further use predictions.
    print("Job done! (No metric can be presented because there is no correct answer given.)")

if __name__ == "__main__":
    main()
