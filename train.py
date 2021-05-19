import logging
import os
import sys
import torch

from datasets import load_dataset, load_from_disk, concatenate_datasets, load_metric
from transformers import DataCollatorWithPadding, EvalPrediction
from transformers import HfArgumentParser

from utils import set_seed, get_MDT_parsers, get_CTM
from utils_qa import tokenize_into_morphs, check_no_error, get_column_names
from utils_qa import preprocess_features_of_Dataset, prepare_train_features, prepare_validation_features
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
    data_args.dataset_path = "/opt/ml/input/data/train_dataset"
    intialize_training_args(training_args) # variable name is training_args, not inference_args.
    
def intialize_training_args(training_args):
    training_args.do_train = True
    training_args.do_predict = False
    training_args.do_eval = False
    training_args.evaluation_strategy = "no"
    if training_args.save_state_only:
        training_args.save_strategy = "no"


def main():
    # SETTINGS
    # arguments and initialization
    model_args, data_args, training_args = get_MDT_parsers()
    initialize(data_args, training_args)
    print(f"data is from {data_args.dataset_path}")
    print(f"model is from {model_args.model_path}")

    # config, tokenizer and model
    config, tokenizer, model = get_CTM(model_args)
    model.train()
    print("model uses device:", model.device)

    # DATA
    # competition datasets
    # type: DatasetDict with key = ["train", "validation"] (standard dataset format for QA)
    datasets = load_from_disk(data_args.dataset_path)
    last_checkpoint, data_args.max_seq_length = check_no_error(training_args, data_args, tokenizer, datasets)
    print(datasets)
    
    train_dataset = datasets["train"]     # type: Dataset
    eval_dataset = datasets["validation"] # type: Dataset
    column_names = get_column_names()

    # train/eval dataset
    # type: Dataset
    if training_args.do_eval: # Run passage retrieval (not implemented)
        retriever = BM25SparseRetriever()
        eval_dataset = retriever.retrieve_standard_dataset_for_QA(eval_dataset, topk=data_args.topk)["validation"] # id, question, context
        eval_dataset = eval_dataset.map(
            lambda example: prepare_validation_features(example, tokenizer, data_args), # tokenize dataset
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else: # use all datasets for training
        datasets_list = [
            preprocess_features_of_Dataset(train_dataset).flatten_indices(),
            preprocess_features_of_Dataset(eval_dataset).flatten_indices(),
        ]
        # KorQuAD v1 dataset
        if data_args.train_korquad:
            korquad_datasets = load_dataset("squad_kor_v1")
            print("korquad dataset:\n", korquad_datasets)

            datasets_list.extend([
                preprocess_features_of_Dataset(korquad_datasets["train"]).flatten_indices(),
                preprocess_features_of_Dataset(korquad_datasets["validation"]).flatten_indices(),
            ])
        train_dataset = concatenate_datasets(datasets_list)
        print("train dataset:\n", train_dataset)

    # train dataset: tokenized (question + context) sequences <- prepare_train_features
    train_dataset = train_dataset.map(
        lambda example: prepare_train_features(example, tokenizer, data_args),
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

    # TRAINING
    # trainer
    print("Init Trainer...")
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,         # tokenized question-context sequence
        eval_dataset=eval_dataset if training_args.do_eval else None,            # tokenized question-context sequence (Option)
        eval_examples=datasets["validation"] if training_args.do_eval else None, # before tokenization; including answers (Option)
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    # checkpoint
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_path):
        checkpoint = model_args.model_path
    else:
        checkpoint = None
    # train
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # SAVE
    if training_args.save_state_only:
        torch.save(model.state_dict(), os.path.join(training_args.output_dir, model_args.model_path + "_state_dict.pth"))
    else:
        trainer.save_model()  # saves the tokenizer too for easy upload

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

        # need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )
    print("model saved.")

    # EVALUATION
    if training_args.do_eval: # not implemented
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
