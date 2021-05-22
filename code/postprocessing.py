from konlpy.tag import Mecab
import json

mecab = Mecab()
with open('/opt/ml/code/outputs/predictions.json') as pr:
    predictions = json.load(pr)
with open('/opt/ml/code/outputs/nbest_predictions.json') as nb:
    nbest_predictions = json.load(nb)


post_predictions = {}
for key, value in predictions.items():
    if value[0] in (' ', '\n', '\xa0'):
        value = value[1:]
    poses = [pos for _, pos in mecab.pos(value)]

    if mecab.pos(value)[-1][-1] in ('JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC'):
        print('끝에 조사', key, value)
        last = mecab.pos(value)[-1][0]
        value = value[:-len(last)]
        print(f'uses {value}')

    if ('EP' in poses or 'EF' in poses or 'EC' in poses) and len(value) > 10:
        print('문장 끝:', len(value), key, value, mecab.pos(value))
        for i in range(1, 20):
            value = nbest_predictions[key][i]['text']
            if value[0] in (' ', '\n', '\xa0'):
                value = value[1:]
            poses = [pos for _, pos in mecab.pos(value)]
            if ('EP' not in poses and 'EF' not in poses and 'EC' not in poses) or len(value) <= 10:
                if mecab.pos(value)[-1][-1] in ('JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC'):
                    print('>>끝에 조사', value)
                    last = mecab.pos(value)[-1][0]
                    value = value[:-len(last)]
                print(f'uses {i}th best: {value}')
                break
            print(f'>>문장 끝 {i}th: {value}')
    post_predictions[key] = value

with open('/opt/ml/code/outputs/post_processing.json', 'w') as fp:
    json.dump(post_predictions, fp)