import pandas as pd

sentiment = pd.read_csv('BoWsentiment.csv')
id_value = dict(zip(sentiment['content_id'], sentiment['sentiment_value']))

submit = pd.read_csv('rcnn.csv')

sentiment_value = []
for content_id in submit['content_id'].tolist():
    sentiment_value.append(id_value[content_id])
submit['sentiment_value'] = sentiment_value
submit.to_csv('n_sentiment.csv', index=False)
