import pandas as pd


polarity = pd.read_csv('result_36polarity.csv')
sentimental = pd.read_csv('result_36sentimental.csv')

sentiment_value = []
for i in range(len(sentimental['sentiment_value'])):
    if sentimental['sentiment_value'][i] == 0:
        sentiment_value.append(0)
    elif polarity['sentiment_value'][i] == 0:
        sentiment_value.append(-1)
    else:
        sentiment_value.append(1)

sentimental['sentiment_value'] = sentiment_value
sentimental.to_csv('BoWsentiment.csv')
