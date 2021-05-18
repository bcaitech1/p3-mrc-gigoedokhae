import os
import re
import json
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from datasets import load_from_disk, concatenate_datasets
from datasets import Sequence, Value, Features, Dataset, DatasetDict

from utils import timer
from utils_qa import tokenize_into_morphs
from bm25 import BM25Vectorizer


class BM25SparseRetriever:
    def __init__(
        self,
        tokenizer=tokenize_into_morphs,
        corpus_path="/opt/ml/input/data/wikipedia_documents.json",
        ):
        self.contexts = self.get_original_contexts(corpus_path)
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        # Transform by vectorizer
        self.vectorizer = BM25Vectorizer(
            tokenizer=tokenizer,
            ngram_range=(1, 2),
            max_features=100000,
            dtype=np.float32,
        )

        self.p_embedding = None
        self._load_sparse_passage_embedding()

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

    def _load_sparse_passage_embedding(self):
        p_embedding_path = "/opt/ml/input/data/sparse_passage_embedding.bin"
        vectorizer_path = "/opt/ml/input/data/bm25_vectorizer.bin"
        if os.path.isfile(p_embedding_path) and os.path.isfile(vectorizer_path):
            with open(p_embedding_path, "rb") as f:
                self.p_embedding = pickle.load(f)

            with open(vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            print("Passage Embedding Loaded.")

        else:
            with timer("Building Passage Embedding..."):
                self.p_embedding = self.vectorizer.fit_transform(self.contexts)
            print(self.p_embedding.shape)

            with open(p_embedding_path, "wb") as f:
                pickle.dump(self.p_embedding, f)
            with open(vectorizer_path, "wb") as f:
                pickle.dump(self.vectorizer, f)
            print("Passage Embedding Saved.")

    def retrieve_standard_dataset_for_QA(self, examples, topk=1):
        with timer("Relevant documents exhaustive search."):
            all_doc_indices = self._get_all_doc_indices(examples["question"], topk=topk)
        dataset = self._get_examples_with_retrieved_context(examples, all_doc_indices)
        return self._get_standard_dataset_for_QA(dataset)

    def _get_all_doc_indices(self, queries, topk=1):
        q_embedding = self.vectorizer.transform(queries)
        
        all_doc_scores = q_embedding * self.p_embedding.T
        if not isinstance(all_doc_scores, np.ndarray):
            all_doc_scores = all_doc_scores.toarray()
        all_doc_indices = []
        for doc_scores in all_doc_scores:
            ranked_indices = np.argsort(doc_scores)[::-1]
            all_doc_indices.append(ranked_indices.tolist()[:topk])
        return all_doc_indices

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
    # Test sparse
    org_datasets = load_from_disk("/opt/ml/input/data/train_dataset")
    dataset = concatenate_datasets(
        [
            org_datasets["train"].flatten_indices(),
            org_datasets["validation"].flatten_indices(),
        ]
    ) # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*"*40, "query dataset", "*"*40)
    print(dataset)

    retriever = BM25SparseRetriever()
    # 정확성 테스트 (미구현)
    # with timer("bulk query by exhaustive search"):
    #     all_doc_indices = retriever._get_all_doc_indices(dataset["question"], topk=1)
    #     df = retriever._get_examples_with_retrieved_context(dataset, all_doc_indices)
    #     df["correct"] = df["original_context"] == df["context"]
    #     print("correct retrieval result by exhaustive search", df["correct"].sum() / len(df))

