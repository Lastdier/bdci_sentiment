import jieba
import pandas as pd


jieba.load_userdict('userdict.txt')
train = pd.read_csv('data/train.csv')
outstr = ''
for content in train['content']:
    temp = ''
    content = content.replace(' ', '')
    content = content.replace('，', ',')
    content = content.replace('？', '?')
    content = content.replace('！', '!')
    for w in jieba.cut(content, cut_all=False):
        temp += w + '\t'
    temp = temp[:-1]
    temp += '\n'
    outstr += temp

out_f = open('segmentation_train.txt', 'w', encoding='utf-8')
out_f.write(outstr)
out_f.close()
