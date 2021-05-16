# PStage 3: Open-Domain Questin Answering

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

## run.ipynb
- training이나 inference를 수행합니다.

## train command 예시
- train: `python train.py --output_dir ./models/train_dataset/debug --do_train`  
- eval : `python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval`

## test command 예시
`python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../input/data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict`

## 설정 및 조작
- 후처리 토크나이저 선택: utils_qa.py > class AnswerPostprocessor > def get_postprocessed_text(tokenizer_name="khaiii" or "mecab")
