# Readme

### Commit 05.10

```
bm25_retrieval.py
inference_presearched.py
utils_qa.py
```
- `bm25_retrieval.py`:
bm25+ 라이브러리 사용 코드입니다.
  
  `initialize()`: wikipedia_documents.json을 호출하여 BM25Plus로 vectorize하여 pickle로 dump합니다.

  `retrieve(dataset, k)`: dump된 bm25 object를 호출하고 dataset을 순회하며 `question`을 vectorize하여 score 순으로 documnet를 `context` 리스트 에 집어넣습니다. `k`에 따라 `context`의 길이가 정해집니다.
  
  `make_test_set()`: retrieve()를 거친 dataframe을 datasets으로 저장합니다.
- `inference_presearched.py`:
  retrieval을 거치지 않고 구해둔 후보 context에서 답을 얻습니다.
  
  145라인에서 `context` list에서 context를 순회하면서 해당 문서만 들어있는 dataset을 재구성합니다. 재구성된 dataset은 `context`와 `context_id`가 있습니다.
  
  `post_processing_function`: `utils_qa.py`의 `postprocess_qa_predictions_top20()`를 호출합니다.
  
  그 외엔 기존 코드와 동일합니다.
- `utils_qa.py`:
  `postprocess_qa_predications_top20()`: 이름은 top20이지만 더 많은 문서, 혹은 하나의 문서여도 가능합니다.
  
  example을 순회하며 `id`, `context_id`를 저장하며 predict하고, 최종적으로 `id` 별로 best score를 구해 해당하는 `context_id`에서 text를 구하여 json에 저장합니다.
