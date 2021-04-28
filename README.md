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

## train
- train: `python train.py --output_dir ./models/train_dataset --do_train`  
- eval : `python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval`

## test
`python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../input/data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict`