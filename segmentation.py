import jieba
import pandas as pd
import re


jieba.load_userdict('userdict.txt')
train = pd.read_csv('data/train_2.csv')
outstr = ''
content_set = set()
for i, content in enumerate(train['content']):
    # temp = train['content_id'][i] + '\t'
    if content in content_set:
        continue
    content_set.add(content)
    temp = ''
    content = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', content)
    content = content.replace(' ', '')
    content = content.replace('，', ',')
    content = content.replace('？', '?')
    content = content.replace('！', '!')
    content = content.replace('（', '(')
    content = content.replace('）', ')')
    for w in jieba.cut(content, cut_all=False):
        temp += w + '\t'
    temp = temp[:-1]
    temp += '\n'
    outstr += temp

out_f = open('segmentation_train.txt', 'w', encoding='utf-8')
out_f.write(outstr)
out_f.close()
