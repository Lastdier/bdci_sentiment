import pandas as pd


train = pd.read_csv('train_2.csv')

rows = list(zip(train['content_id'].tolist(), train['content'].tolist(), train['subject'].tolist(), train['sentiment_value'].tolist()))
temp = []
for r in rows:
    if r[2] == "舒适性":
        temp.append(r)
content_id, content, subject, sentiment_value = zip(*temp)
outf = pd.DataFrame({'content_id': content_id,
                    'content': content,
                    'subject': subject,
                    'sentiment_value': sentiment_value})
outf.to_csv('shushi.csv', index=None)
