import jieba
import pandas as pd


jieba.load_userdict('userdict.txt')
train = pd.read_csv('data/train.csv')
outstr = ''
for i, content in enumerate(train['content']):
    temp = train['content_id'][i] + '\t'
    content = content.replace(' ', '')
    content = content.replace('，', ',')
    content = content.replace('？', '?')
    content = content.replace('！', '!')
    for w in jieba.cut(content, cut_all=False):
        temp += w + '\t'
    temp = temp[:-1]
    temp += '\n'
    outstr += temp

out_f = open('segmentation_train_id.txt', 'w', encoding='utf-8')
out_f.write(outstr)
out_f.close()
