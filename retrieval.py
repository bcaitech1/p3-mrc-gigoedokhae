import os
import re
import json
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
from datasets import load_from_disk
from datasets import Sequence, Value, Features, Dataset, DatasetDict

from utils import timer
# from utils_qa import tokenize_into_morphs


class DenseRetriever:
    def __init__(
        self,
        tokenizer,
        context_encoder,
        question_encoder,
        corpus_path="/opt/ml/input/data/wikipedia_documents.json",
        contexts_embedding_path = "/opt/ml/input/data/dense_passage_embedding.bin",
        questions_embedding_path = "/opt/ml/input/data/dense_question_embedding.bin",
        ):
        self.contexts = self.get_original_contexts(corpus_path)
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        self.contexts_embedding = self._get_dense_embedding(
            contexts_embedding_path,
            context_encoder,
            self.contexts
        )
        self.question_encoder = question_encoder
        self.questions_embedding_path = questions_embedding_path
        self.questions_embedding = None

    def get_original_contexts(self, corpus_path):
        # 중복 시 제거, 한글 미포함 시 제거
        with open(corpus_path, "r") as f:
            corpus = json.load(f)
        unique_contexts = dict.fromkeys([v["text"] for v in corpus.values()]) # list comprehension 순서 보존
        korean_contexts = []
        for context in unique_contexts:
            if self._is_korean_in(context):
                korean_contexts.append(context)
        return korean_contexts

    def _is_korean_in(self, context):
        return re.search("[가-힇]", context)

    def _get_dense_embedding(self, path, encoder, texts):
        if os.path.isfile(embedding_path):
            with open(embedding_path, "rb") as f:
                embedding = pickle.load(f)
            print("Dense Embedding Loaded.")
        else:
            with timer("Building Dense Embedding..."):
                with torch.no_grad():
                    encoder.eval()
                    tokenized_texts = tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
                    embedding = encoder(**tokenized_texts).to("cpu") # why to cpu? / to("cpu") 대신 detach() ?
            print(embedding.shape)

            with open(embedding_path, "wb") as f:
                pickle.dump(embedding, f)
            print("Passage Embedding Saved.")

    def retrieve_standard_dataset_for_QA(self, examples, topk=1):
        with timer("Relevant documents exhaustive search."):
            all_doc_indices = self._get_all_doc_indices(examples["question"], topk=topk)
        dataset = self._get_examples_with_retrieved_context(examples, all_doc_indices)
        return self._get_standard_dataset_for_QA(dataset)

    def _get_all_doc_indices(self, queries, topk=1):
        self.questions_embedding = self._get_dense_embedding(
            self.questions_embedding_path,
            self.question_encoder,
            queries
        )
        dot_prod_scores = torch.matmul(self.questions_embedding, torch.transpose(self.contexts_embedding, 0, 1))
        sorted_doc_indices = torch.argsort(dot_prod_scores, dim=1, decsending=True).squeeze()
        return sorted_doc_indices[:topk]

    def _get_examples_with_retrieved_context(self, examples, all_doc_indices):
        dataset = []
        for i, (example, doc_indices) in enumerate(tqdm(zip(examples, all_doc_indices), desc="Making DataFrame dataset:")):
            for j, doc_index in enumerate(doc_indices):
                data = {
                    "id": example["id"] + str(j),
                    "question": example["question"],
                    "context": self.contexts[doc_index], # retrieved doument
                }
                if "context" in example.keys() and "answers" in example.keys():
                    data["original_context"] = example["context"]  # original document
                    data["answers"] = example["answers"]           # original answer
                dataset.append(data)
        return dataset

    def _get_standard_dataset_for_QA(self, dataset):
        df = pd.DataFrame(dataset)
        features = Features({
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
            "context": Value(dtype="string", id=None),
        })
        return DatasetDict({
            "validation": Dataset.from_pandas(df, features=features)
        })


if __name__ == "__main__":
    pass
    # Test sparse
    # org_datasets = load_from_disk("/opt/ml/input/data/train_dataset")
    # dataset = concatenate_datasets(
    #     [
    #         org_datasets["train"].flatten_indices(),
    #         org_datasets["validation"].flatten_indices(),
    #     ]
    # ) # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    # print("*"*40, "query dataset", "*"*40)
    # print(dataset)

    # retriever = DPRRetriever()
    # 정확성 테스트 (미구현)
    # with timer("bulk query by exhaustive search"):
    #     all_doc_indices = retriever._get_all_doc_indices(dataset["question"], topk=1)
    #     df = retriever._get_examples_with_retrieved_context(dataset, all_doc_indices)
    #     df["correct"] = df["original_context"] == df["context"]
    #     print("correct retrieval result by exhaustive search", df["correct"].sum() / len(df))

