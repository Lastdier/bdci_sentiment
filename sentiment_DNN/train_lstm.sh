python main_sentiment_cv.py main --max_epoch=30 --model='Multitask' --batch-size=1  --lr=0.001 --lr2=0.000 --lr_decay=0.8 --content-dim=500  --type_='word'  --kernel-size=2 --kmax-pooling=1 --linear-hidden-size=500 --hidden-size=256 --num-workers=4 --embedding_path='sgns.weibo.bigram-char' 