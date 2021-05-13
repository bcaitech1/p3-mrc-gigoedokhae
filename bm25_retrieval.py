import faiss
from rank_bm25 import BM25Plus

from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from konlpy.tag import Mecab

import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')


class Bm25SparseRetrieval:
    def __init__(self, tokenize_fn, data_path="../input/data/", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        # wikipedia_documents는 '인덱스 번호'를 key 값으로 갖고 dict를 value로 갖는 json 파일이다.(60613개)
        # 각 dict는 text, corpus_source, url, domain, title, author, html, document_id를 key 값으로 갖는다.
        # wikipedia_documents에서 context에 해당하는 text 부분을 받아와서 리스트화 한다.
        self.contexts = list(dict.fromkeys(
            set(v['text'] for v in wiki.values())))  # set 은 매번 순서가 바뀌므로(list(dict)는 key값으로 리스트를 만든다.)
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # context로 학습된 bm25 객체 생성 및 pickle로 저장
        self.bm25 = initialize_bm25_object()


        # should run get_sparse_embedding() or build_faiss() first.
        self.bm25v = None
        self.p_embedding = None
        self.indexer = None
        
    def initialize_bm25_object():
        # class initializer의 tokenize_fn은 형태소 기반인 mecab.morphs(text)이다.
        tokenized_contexts = [tokenize_fn(context) for context in self.contexts]
        bm25 = BM25Plus(tokenized_contexts)
        # pickle로 bm25 객체 저장
        with open('bm25_object.pickle', 'wb') as f:
            pickle.dump(bm25, f)
        return bm25
    
    def get_sparse_embedding(self):
        # Pickle save.
        # /input/data/에는 dummy_dataset, test_dataset, train_dataset, wikipedia_document 존재
        emd_path = os.path.join(self.data_path, "sparse_embedding.bin")
        bm25v_path = os.path.join(self.data_path, "bm25v.bin")
        
        # 만들어둔게 존재하면 불러오고, 아니면 새로 만든다.
        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)  # 기존에 만든 sparse embedding 벡터를 불러온다.
            with open(bm25v_path, "rb") as file:
                self.bm25v = pickle.load(file)
            print("Embedding pickle loaded👍")
        else:
            print("Building passage embedding...🤔")
            self.p_embedding = self.tfidfv.fit_transform(
                self.contexts)  # 문서 전체에 해당하는 term들을 정의하고 idf도 계산
            print(f'P embedding shape: {self.p_embedding.shape}')
            
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.bm25v, file)
            print("Embedding pickle saved👍")

    def retrieve(self, dataset, topk=1):
        assert self.p_embedding is not None, "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
        assert isinstance(dataset, Dataset), "dataset is not instance of Dataset👾"

        # make retrieved result as dataframe
        total = []
        for data in tqdm(dataset, desc='BM25 retrieval processing👾'):
            scores = self.bm25.get_scores(tokenize(data['question']))
            topk_indices = np.argsort(scores)[::-1][:topk]  # 내림차순 정렬해서 topk개의 인덱스 추출
            topk_retrieved = [self.contexts[i] for i in topk_indices]  # 추출한 인덱스에 해당하는 context 뽑아오기

            tmp = {
                "question": data["question"],
                "id": data['id'],
                "context_id": topk_indices,  # retrieved id
                "context": topk_retrieved  # retrieved document
            }

            if 'context' in data.keys() and 'answers' in data.keys():
                tmp["original_context"] = data['context']  # original document
                tmp["answers"] = data['answers']  # original answer
            total.append(tmp)

        cqas = pd.DataFrame(total)
        return cqas

    def get_relevant_doc(self, query, k=1):
        """
        참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=1):
        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

if __name__ == "__main__":
    # Test sparse
    org_dataset = load_from_disk("../input/data/data/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*"*40, "query dataset", "*"*40)
    print(full_ds)

    ### Mecab 이 가장 높은 성능을 보였기에 mecab 으로 선택 했습니다 ###
    mecab = Mecab()

    def tokenize(text):
        return mecab.morphs(text)

    wiki_path = "wikipedia_documents.json"
    retriever = SparseRetrieval(
        tokenize_fn=tokenize,
        data_path="data",
        context_path=wiki_path)

    # test single query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("single query by exhaustive search"):
        scores, indices = retriever.retrieve(query)
    with timer("single query by faiss"):
        scores, indices = retriever.retrieve_faiss(query)

    # test bulk
    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve(full_ds)
        df['correct'] = df['original_context'] == df['context']
        print("correct retrieval result by exhaustive search",
              df['correct'].sum() / len(df))
    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve_faiss(full_ds)
        df['correct'] = df['original_context'] == df['context']
        print("correct retrieval result by faiss",
              df['correct'].sum() / len(df))
