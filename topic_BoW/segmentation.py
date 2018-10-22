import jieba
import pandas as pd


train = pd.read_csv('data/test_public.csv')
outstr = ''
for content in train['content']:
    temp = ''
    content = content.replace('，', ',')
    content = content.replace('？', '?')
    content = content.replace('！', '!')
    for w in jieba.cut(content, cut_all=False):
        if w == ' ':
            continue
        temp += w + '\t'
    temp = temp[:-1]
    temp += '\n'
    outstr += temp

out_f = open('segmentation_test.txt', 'w')
out_f.write(outstr)
out_f.close()
