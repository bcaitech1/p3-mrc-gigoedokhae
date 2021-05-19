import logging
import os
import torch
from transformers import HfArgumentParser
from transformers import AutoConfig, AutoTokenizer
from custom_models import CustomXLMRoberta
from arguments import ModelArguments, DataTrainingArguments, TrainingArguments

import time
from contextlib import contextmanager

import random
import torch
import numpy as np

import re
from glob import glob
from pathlib import Path



from korbert.src_tokenizer.tokenization_morp import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering, BertConfig



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
    model_name = "bert-large-cased"
    model_path = "/opt/ml/code/korbert/pytorch_model.bin"
    config_path = "/opt/ml/code/korbert/bert_config.json"

    config = BertConfig(config_path)
    model = BertForQuestionAnswering(config)
    state_dict = torch.load(os.path.join(model_name, model_path))

    del_list = ["cls.predictions.bias", "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias", "cls.predictions.decoder.weight", "cls.seq_relationship.weight", "cls.seq_relationship.bias"]
    for elt in del_list:
        state_dict.pop(elt)
    state_dict["qa_outputs.weight"] = torch.zeros([2, 768])
    state_dict["qa_outputs.bias"] = torch.zeros([2])

    model.load_state_dict(state_dict)
    model.to("cuda")

    vocab_path = "/opt/ml/code/korbert/vocab.korean_morp.list"
    tokenizer_path = os.path.join(model_name, vocab_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)



    # config = AutoConfig.from_pretrained(
    #     model_args.config_name
    #     if model_args.config_name
    #     else model_args.model_path
    # )

    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.tokenizer_name
    #     if model_args.tokenizer_name
    #     else model_args.model_path,
    #     use_fast=True
    # )

    # if model_args.model_state_path == "no"\
    # or not os.path.isdir(model_args.model_state_path): # "is None" not works.
    #     logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    #     model = CustomXLMRoberta.from_pretrained(
    #         model_args.model_path,
    #         config=config,
    #     )
    # else: # using model state
    #     model = CustomXLMRoberta.from_config(config)
    #     model.load_state_dict(torch.load(model_args.model_state_path))
    # model.to("cuda")

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
    """
    - increment_path("/root", "/no", "test") -> "/root/no1_test"
    - increment_path("/root/test", "_no", "prediction.json") -> "/root/test_no1/prediction.json"
    """
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
