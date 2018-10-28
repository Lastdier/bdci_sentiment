from sklearn.externals import joblib
import pandas as pd
from sklearn.metrics import f1_score


polarity = joblib.load('y_pred_polarity.pk')
polarity_id = pd.read_csv('../data/multilabel_polarity.csv')['id']
polarity_dict = {}
for i, content_id in enumerate(polarity_id):
    polarity_dict[content_id] = polarity[i] 

sentimental = joblib.load('y_pred_sentimental.pk')
sentimental_id = pd.read_csv('../data/multilabel_issentimental.csv')['id']
sentimental_dict = {}
for i, content_id in enumerate(sentimental_id):
    sentimental_dict[content_id] = sentimental[i] 

train = pd.read_csv('../data/train_2.csv')
predict = []
for i, content_id in enumerate(train['content_id']):
    if sentimental_dict[content_id] == 0:
        predict.append(0)
    elif polarity_dict.get(content_id) is None:
        predict.append(0)
    elif polarity_dict[content_id] == 0:
        predict.append(-1)
    elif polarity_dict[content_id] == 1:
        predict.append(1)
true_target = train['sentiment_value'].tolist()
f1 = f1_score(true_target, predict, average='micro')
print('train_f1: %f' % f1)

