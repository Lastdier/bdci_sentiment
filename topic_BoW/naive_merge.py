import sys
sys.path.append("..")
from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score
from time import time
import numpy as np
import scipy.sparse as sp
import pandas as pd

from utils.path_util import from_project_root
from utils.proba_util import predict_proba
from term_weighting_model.transformer import generate_vectors
from term_weighting_model.stacker import generate_meta_feature, gen_data_for_stacking
from term_weighting_model.stacker import model_stacking_from_pk


SUBJECT_LIST = ['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观']
def main(save=False):
    # load data from pickle
    pk_url = from_project_root("processed_data/vector/36_dc_idf_xgb.pk")
    print("loading data from", pk_url)
    X, y, X_test = joblib.load(pk_url)

    pk_url = from_project_root("rcnn.pk")
    print("loading data from", pk_url)
    X2, X_test2 = joblib.load(pk_url)

    pk_url = from_project_root("cnn.pk")
    print("loading data from", pk_url)
    X3, X_test3 = joblib.load(pk_url)
#
    #train1 = pd.read_csv('../cf/xgb_multilabel_tfbdc_train.txt', sep=' ', header=None).values
    #train2 = pd.read_csv('../cf/xgb_multilabel_tficf_train.txt', sep=' ', header=None).values
    #train3 = pd.read_csv('../cf/xgb_multilabel_tfidf_train.txt', sep=' ', header=None).values
    #train4 = pd.read_csv('../cf/lgbm_multilabel_tfbdc_train.txt', sep=' ', header=None).values
    #train5 = pd.read_csv('../cf/lgbm_multilabel_tficf_train.txt', sep=' ', header=None).values
    #train6 = pd.read_csv('../cf/lgbm_multilabel_tfbdc_train.txt', sep=' ', header=None).values
    #X = np.concatenate((X, X2), axis=1)
    #
#
    #test1 = pd.read_csv('cf/xgb_multilabel_tfbdc_test.txt', sep=' ', header=None).values
    #test2 = pd.read_csv('cf/xgb_multilabel_tficf_test.txt', sep=' ', header=None).values
    #test3 = pd.read_csv('cf/xgb_multilabel_tfidf_test.txt', sep=' ', header=None).values
    #test4 = pd.read_csv('cf/lgbm_multilabel_tfbdc_test.txt', sep=' ', header=None).values
    #test5 = pd.read_csv('cf/lgbm_multilabel_tficf_test.txt', sep=' ', header=None).values
    #test6 = pd.read_csv('cf/lgbm_multilabel_tfidf_test.txt', sep=' ', header=None).values
    #X_test = np.concatenate((X_test, X_test2), axis=1)

    test_url = from_project_root("data/test_processed.csv")

    print(X.shape, y.shape, X_test.shape)
    
    # result = X_test * 0.9 + (test1 + test2 + test3 + test4 + test5 + test6) / 60
    result = X_test3
    
    test_public = pd.read_csv(test_url)
    output_str = 'content_id,subject,sentiment_value,sentiment_word\n'
    for jjj in range(len(result)):
        
        aaa = np.arange(10)
        labels = aaa[result[jjj]>0.5]

        if len(labels) == 0:
            labels = result[jjj].argmax()    # 选择有概率最高的作为分类
            #output_str += "%s,%s,0,\n" % (test_public['id'][jjj], '无')     # 留出无分类的用单标签分类模型分类
            #continue

        if type(labels) == np.int64:
            output_str += "%s,%s,0,\n" % (test_public['id'][jjj], SUBJECT_LIST[labels])
            continue

        for kkk in labels:
            output_str += "%s,%s,0,\n" % (test_public['id'][jjj], SUBJECT_LIST[kkk])
    outfile = open('resultrcnn.csv', 'w')
    outfile.write(output_str)
    outfile.close()


if __name__ == '__main__':
    main()
