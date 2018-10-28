import json


class Config(object):
    loss = 'crossEntropy'
    model = 'MultiCNNTextBNDeep'
    content_dim = 300
    embedding_dim = 300
    linear_hidden_size = 2000
    num_classes = 3
    
    type_ = 'word'

    word2index_path = 'word2index.json'

    fff = open('word2index.json', 'r')
    word2index = json.load(fff)
    fff.close()
    f = open('char2index.json', 'r')
    char2index = json.load(f)
    f.close()

    word_size = len(word2index)
    char_size = len(char2index)
    vocab_size = word_size

    seq_len = 100
    if type_ == 'char':
        vocab_size = char_size
        seq_len = 150

    kernel_size = 3
    kernal_sizes = [1,2,3,4]
   # 95%_len = 76

    embedding_path = 'sgns.weibo.bigram-char'

    max_epoch=30
    lr = 5e-4 # 学习率
    lr2 = 1e-4 # embedding层的学习率
    min_lr = 1e-5
    lr_decay = 0.8
    weight_decay = 0

    batch_size = 128
    plot_every = 10
    decay_every = 40
    augument = False

    static = False
    num_workers = 10

    kmax_pooling = 1
    hidden_size = 300
    num_layers = 1

    early_stoping = 5


def parse(self,kwargs,print_=True):

        for k,v in kwargs.items():
            if not hasattr(self,k):
                raise Exception("opt has not attribute <%s>" %k)
            setattr(self,k,v) 
        
        return self

Config.parse = parse
opt = Config()

