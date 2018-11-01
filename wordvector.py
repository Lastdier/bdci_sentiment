from gensim.models import Word2Vec
#import jieba
import re
import pandas as pd
from pyhanlp import *


# jieba.load_userdict('userdict.txt')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test_public.csv')
outstr = ''
content_set = set()
for i, content in enumerate(train['content']):
    # temp = train['content_id'][i] + '\t'
    if content in content_set:
        continue
    content_set.add(content)
    content = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', content)
    content = content.replace(' ', '')
    content = content.replace('，', ',')
    content = content.replace('？', '?')
    content = content.replace('！', '!')
    content = content.replace('（', '(')
    content = content.replace('）', ')')
    content_set.add(content)
for i, content in enumerate(test['content']):
    # temp = train['content_id'][i] + '\t'
    if content in content_set:
        continue
    content_set.add(content)
    content = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', content)
    content = content.replace(' ', '')
    content = content.replace('，', ',')
    content = content.replace('？', '?')
    content = content.replace('！', '!')
    content = content.replace('（', '(')
    content = content.replace('）', ')')
    content_set.add(content)

content_list = []
char_list = []
outstr = ''
charstr = ''
for content in content_set:
    this_list = HanLP.segment(content)
    temp = []
    for w in this_list:
        w = w.word
        temp.append(w)
        outstr += w + ' '
    outstr = outstr[:-1]
    outstr += '\n'
    content_list.append(temp)
    temp = []
    for c in content:
        temp.append(c)
        charstr += c + ' '
    char_list.append(temp)
    charstr = charstr[:-1]
    charstr += '\n'
# outfile = open('corpus_word.txt', 'w', encoding='utf-8')
# outfile.write(outstr)
# outfile.close()
# outfile = open('corpus_char.txt', 'w', encoding='utf-8')
# outfile.write(charstr)
# outfile.close()
model = Word2Vec(content_list, size=100, window=5, min_count=1, workers=10, iter=10)
model.wv.save_word2vec_format('myword_vector.100dim', binary=False)
model = Word2Vec(char_list, size=100, window=5, min_count=1, workers=10, iter=10)
model.wv.save_word2vec_format('mychar_vector.100dim', binary=False)
