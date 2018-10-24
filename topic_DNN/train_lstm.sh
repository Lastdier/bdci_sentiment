python main_cv.py main --weight_decay=0 --max_epoch=30 --model='LSTMwithAtten' --batch-size=512  --lr=0.001 --lr2=0.000 --lr_decay=0.8 --content-dim=500  --type_='word'  --kernel-size=2 --kmax-pooling=1 --linear-hidden-size=2000 --hidden-size=256 --num-workers=4 --embedding_path='sgns.weibo.bigram-char' 

