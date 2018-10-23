import json


class Config(object):
    loss = 'multilabelloss'
    model = 'MultiCNNTextBNDeep'
    content_dim = 300
    num_class = 10
    embedding_dim = 300
    linear_hidden_size = 500
    num_classes = 10
    
    type_ = 'word'

    word2index_path = 'word2index.json'

    fff = open(word2index_path, 'r')
    word2index = json.load(fff)
    fff.close()

    vocab_size = 2867       # len(word2index)

    if type_ == 'word':
        vocab_size = len(word2index)

    kernel_size = 3
    kernal_sizes = [1,2,3,4]

    seq_len = 150   # 95%_len = 76

    embedding_path = 'sgns.weibo.bigram-char'

    max_epoch=30
    lr = 5e-4 # 学习率
    lr2 = 1e-4 # embedding层的学习率
    min_lr = 1e-5
    lr_decay = 0.99
    weight_decay = 1e-5

    batch_size = 128
    plot_every = 10
    decay_every = 40
    augument = False

    static = False
    num_workers = 10

    kmax_pooling = 1
    hidden_size = 300
    num_layers = 2

    early_stoping = 5


def parse(self,kwargs,print_=True):

        for k,v in kwargs.items():
            if not hasattr(self,k):
                raise Exception("opt has not attribute <%s>" %k)
            setattr(self,k,v) 
        
        return self

Config.parse = parse
opt = Config()

