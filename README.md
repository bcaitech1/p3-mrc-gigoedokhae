# PStage 3: Open-Domain Question Answering
Extractive, Sparse Passage Retrieval

## install
- data (51.2 MB)  
`tar -xzf data.tar.gz`
- Python packages  
`bash ./install/install_requirements.sh`
- Mecab  
root 계정: `bash ./install/install_mecab_root.sh`  
일반 계정: `bash ./install/install_mecab.sh`  

NameError: name 'Tagger' is not defined 해결
- `apt install curl`
- `bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)`

## Jupyter notebooks
- data_preprocessing.ipynb: 데이터 전처리를 수행합니다.
- run.ipynb: 학습이나 추론을 수행합니다.
- loop_inference.ipynb: run을 통한 학습 후 실행하여, 모델이 저장될 때마다 추론을 수행합니다. (save_total_limit >= 2)
- ensemble.ipynb: 앙상블을 수행합니다.

## command를 통한 실행 예시
- train: `python train.py --output_dir /opt/ml/outputs/models/debug`  
- infer: `python inference.py --output_dir /opt/ml/outputs/preds/last --topk 40`

## 설정
- 주요 parser 설정(run.ipynb)
  - 공통: exp_name 및 output_dir / model_path 및 model_state_path(if load pretrained model as state dict)
  - train: train_korquad / save_state_only / num_train_epochs 및 learning_rate(transfer-learning > fine-tuning)
  - infer: topk / tokenizer_name(if "Can't load tokenizer")
- 커스텀 모델의 freeze 상태: custom_models.py > class CustomXLMRoberta > def __init__
- 후처리 토크나이저 선택: utils_qa.py > class AnswerPostprocessor > def get_postprocessed_text
