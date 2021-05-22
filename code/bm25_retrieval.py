from rank_bm25 import BM25Plus
from konlpy.tag import Mecab
from tqdm.auto import tqdm
from contextlib import contextmanager
import numpy as np
import pandas as pd
import json
import pickle
import time


from datasets import (Dataset, DatasetDict, load_from_disk)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')

mecab = Mecab()

def tokenize(sentence):
    return mecab.morphs(sentence)

def initialize():
    with open('/opt/ml/input/data/data/wikipedia_documents.json') as fp:
        wiki_data = json.load(fp)
    tokenized = []
    for d in range(len(wiki_data)):
        tokenized.append(tokenize(wiki_data[str(d)]['text']))
    bm25 = BM25Plus(tokenized)

    with open('bm25_mecab_wiki_data.pickle', 'wb') as f:
        pickle.dump(bm25, f)

def load_bm25_pickle(path):
    with open(path, 'rb') as fp:
        bm25 = pickle.load(fp)
    return bm25

def load_wiki(path):
    with open(path) as fp:
        load_data = json.load(fp)
    text_corpus = [load_data[str(idx)]['text'] for idx in range(len(load_data))]
    return text_corpus

def retrieve(dataset, k):
    data_path = '/opt/ml/code/bm25_khaiii_wiki_data.pickle'
    wiki_bm25 = load_bm25_pickle(data_path)
    original_wiki = load_wiki('/opt/ml/input/data/data/wikipedia_documents.json')
    if isinstance(dataset, Dataset):
        total = []
        for idx, example in enumerate(tqdm(dataset, desc='BM25 retrieval: ')):
            scores = wiki_bm25.get_scores(tokenize(example['question']))
            top_n = np.argsort(scores)[::-1][:k]
            context = [original_wiki[i] for i in top_n]

            tmp = {
                "question": example["question"],
                "id": example['id'],
                "context_id": top_n,  # retrieved id
                "context": context  # retrieved doument
            }

            if 'context' in example.keys() and 'answers' in example.keys():
                tmp["original_context"] = example['context']  # original document
                tmp["answers"] = example['answers']  # original answer
            total.append(tmp)
    cqas = pd.DataFrame(total)
    return cqas

def retriever_eval():
    train_dataset = load_from_disk('/opt/ml/input/data/data/train_dataset')
    eval_dataset = train_dataset['validation']
    df = retrieve(eval_dataset, 100)

    datasets = DatasetDict({'validation': Dataset.from_pandas(df)})
    acc = 0
    for original, ret in zip(df['original_context'], df['context']):
        if original in ret:
            acc += 1
    print(acc/len(df))
    return acc / len(df)


def make_test_set():
    test_dataset = load_from_disk('/opt/ml/input/data/data/test_dataset')
    dataset = test_dataset['validation']
    df = retrieve(dataset, 20)

    data = DatasetDict({'validation': Dataset.from_pandas(df)})
    data.save_to_disk('/opt/ml/test_bm25plus_khaiii_top20')

if __name__ == '__main__':
    initialize()
    make_test_set()
