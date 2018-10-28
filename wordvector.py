from gensim import models


model = models.KeyedVectors.load_word2vec_format('word_50_w5.bin', binary=False)
model.save_word2vec_format('word_50_w5.vec', binary=False)
