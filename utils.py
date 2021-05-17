import logging
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
from transformers import HfArgumentParser
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments

import time
from contextlib import contextmanager

import random
import torch
import numpy as np

import re
from glob import glob
from pathlib import Path

__all__ = [
    "get_MDT_parsers", "get_CTM",
    "timer", "set_seed", "increment_path",
]


# Get parsers fpr model, data, training
def get_MDT_parsers():
    """ModelArguments, DataTrainingArguments, TrainingArguments"""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()

# Get config, tokenizer, model
def get_CTM(model_args):
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name,
        use_fast=True
    )

    if model_args.model_state_dir is None:
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name,
            from_tf=bool(".ckpt" in model_args.model_name),
            config=config,
        )
    else: # using model state
        model = AutoModelForQuestionAnswering(config)
        model.load_state_dict(torch.load(model_args.model_state_dir))
    model.to("cuda")

    return config, tokenizer, model


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def increment_path(path, infix="", name=""):
    path += infix
    dirs = glob(f"{path}*")
    stem = Path(path).stem
    matches = [re.search(rf"{stem}(\d+)", d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 1
    return "".join([path, f"{n}_", name])


if __name__ == "__main__":
    # 작동 확인
    model_args, data_args, training_args = get_MDT_parsers()
    config, tokenizer, model = get_CTM(model_args)
