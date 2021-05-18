from typing import Any, Dict, List, Optional
from dataclasses import asdict, dataclass, field
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_path: str = field(
        default="xlm-roberta-large",#bert-base-multilingual-cased",
        metadata={"help": "Pretrained model identifier from huggingface.co/models"}
    )
    model_state_path: Optional[str] = field(
        default="no",
        metadata={"help": "Path to pretrained model state"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_path"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_path"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_path: Optional[str] = field(
        default="/opt/ml/input/data/train_dataset",
        metadata={"help": "The name of the dataset to use."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    train_korquad: bool = field(
        default=False,
        metadata={"help": "Whether to use koquad v1 datasets for training"},
    )

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="/opt/ml/outputs/debug",
        metadata={"help": "Path to save model after training."}
    )
    save_state_only: bool = field(
        default=False,
        metadata={"help": "Whether to save model state only."}
    )
    topk: int = field(
        default=1,
        metadata={"help": "The number of documents to retrieve for each question."},
    )