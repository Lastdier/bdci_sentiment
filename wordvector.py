from gensim.models import Word2Vec
import jieba
import re
import pandas as pd


jieba.load_userdict('userdict.txt')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test_public.csv')
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
    content_set.add(content)
for i, content in enumerate(test['content']):
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
    content_set.add(content)

content_list = []
for content in content_set:
    this_list = jieba.lcut(content, cut_all=False)
    content_list.append(this_list)

model = Word2Vec(content_list, size=100, window=5, min_count=1, workers=10, iter=10)
model.wv.save_word2vec_format('myword_vector.100dim', binary=False)
