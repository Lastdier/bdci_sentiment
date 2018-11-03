# coding: utf-8
# created by deng on 7/27/2018
import sys
sys.path.append("..")
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from time import time
import numpy as np
import scipy.sparse as sp
import pandas as pd

from utils.path_util import from_project_root
from utils.proba_util import predict_proba
from term_weighting_model.transformer import generate_vectors
from term_weighting_model.stacker import generate_meta_feature, gen_data_for_stacking
from term_weighting_model.stacker import model_stacking_from_pk

N_JOBS = 10
N_CLASSES = 10
RANDOM_STATE = 19950717
CV = 5


def train_and_gen_result(clf, X, y, X_test, use_proba=False, save_url=None, n_splits=1, random_state=None):
    """ train and generate result with specific clf

    Args:
        clf: classifier
        X: vectorized data
        y: target
        X_test: test data
        use_proba: predict probabilities of labels instead of label
        save_url: url to save the result file
        n_splits: n_splits for K-fold, None to not use k-fold
        random_state: random_state for 5-fold

    """
    if n_splits > 1:
        slf = StratifiedKFold(n_splits=n_splits, shuffle=bool(random_state), random_state=random_state)
        y_pred_proba = np.zeros((X_test.shape[0], N_CLASSES))
        for train_index, cv_index in slf.split(X, np.zeros((len(y),))):
            X_train = X[train_index]
            y_train = y[train_index]
            clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_test)
            #print(y_pred)
            y_pred_proba += predict_proba(clf, X_test, X_train, y_train)
        y_pred_proba /= n_splits
        y_pred = y_pred_proba.argmax(axis=1) + 1
        # 正确生成多标签？并正确评价多标签
        

    else:
        clf.fit(X, y)
        y_pred_proba = predict_proba(clf, X_test, X, y)
        y_pred = clf.predict(X_test)

    if use_proba:
        result_df = pd.DataFrame(y_pred_proba, columns=['class_prob_' + str(i + 1) for i in range(N_CLASSES)])
    else:
        result_df = pd.DataFrame(y_pred, columns=['class'])
    if save_url:
        result_df.to_csv(save_url, index_label='id')
    return result_df


def get_result_from_stacking(clf, X, y, X_test):
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    print(y_pred.shape)
    #for jjj in y_pred:
    #    aaa = np.arange(10)
    #    print(aaa[jjj==1])
    return y_pred


def val(X, y, X_test):
    predict = []
    for l in X:
        temp = [0] * 10
        temp = np.array(temp)
        temp[l>0.5] = 1
        predict.append(temp)
    predict = np.array(predict)
    f1 = f1_score(y, predict, average='micro')
    print(f1)
    # 计算各类别的f1
    f1_list = []
    percision_list = []
    recall_list = []
    for i in range(10):
        f1_list.append(f1_score(y[:,i], predict[:,i]))
        percision_list.append(precision_score(y[:,i], predict[:,i]))
        recall_list.append(recall_score(y[:,i], predict[:,i]))
    print('价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观')
    print(percision_list)
    print(recall_list)
    print(f1_list)
    


SUBJECT_MASK = {'价格': 0, '配置': 1, '操控': 2, '舒适性': 3, '油耗': 4, '动力': 5, '内饰': 6, '安全性': 7, '空间': 8, '外观': 9}
SUBJECT_LIST = ['价格', '配置', '操控', '舒适性', '油耗', '动力', '内饰', '安全性', '空间', '外观']
def main(save=False):
    # load data from pickle
    pk_url = from_project_root("processed_data/vector/stacked_dc_idf_36.pk")
    print("loading data from", pk_url)
    X, y, X_test = joblib.load(pk_url)

    pk_url = from_project_root("rcnn_word.pk")
    print("loading data from", pk_url)
    X2, X_test2 = joblib.load(pk_url)
    val(X2, y, X_test2)
    pk_url = from_project_root("cnn_word.pk")
    print("loading data from", pk_url)
    X3, X_test3 = joblib.load(pk_url)
    val(X3, y, X_test3)
    pk_url = from_project_root("lstm_word.pk")
    print("loading data from", pk_url)
    X4, X_test4 = joblib.load(pk_url)
    val(X4, y, X_test4)
    pk_url = from_project_root("rcnn_char.pk")
    print("loading data from", pk_url)
    X5, X_test5 = joblib.load(pk_url)
    val(X5, y, X_test5)
    pk_url = from_project_root("cnn_char_cwe.pk")
    print("loading data from", pk_url)
    X6, X_test6 = joblib.load(pk_url)
    val(X6, y, X_test6)
    pk_url = from_project_root("lstm_char.pk")
    print("loading data from", pk_url)
    X7, X_test7 = joblib.load(pk_url)
    val(X7, y, X_test7)
    
    # pk_url = from_project_root("32_hybrid_10_clf.pkl")
    # print("loading data from", pk_url)
    # X8,_, X_test8 = joblib.load(pk_url)
    # val(X8, y, X_test8)
    test_url = from_project_root("data/test_processed.csv")
    
    clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1))  # xgb's default n_jobs=1

    # 与36一同stacking
    # y_pred_proba, y, y_test_pred_proba = gen_data_for_stacking(clf, X, y, X_test, n_splits=5, random_state=RANDOM_STATE)
    # X = np.concatenate((X5[:,0].reshape(-1,1),X6[:,1].reshape(-1,1),y_pred_proba[:,2].reshape(-1,1),X7[:,3].reshape(-1,1),X3[:,4].reshape(-1,1), X5[:,5].reshape(-1,1), X3[:,6].reshape(-1,1), X7[:,7].reshape(-1,1), X7[:,8].reshape(-1,1), X6[:,9].reshape(-1,1)), axis=1)
    # X_test = np.concatenate((X_test5[:,0].reshape(-1,1),X_test6[:,1].reshape(-1,1),y_test_pred_proba[:,2].reshape(-1,1),X_test7[:,3].reshape(-1,1),X_test3[:,4].reshape(-1,1), X_test5[:,5].reshape(-1,1), X_test3[:,6].reshape(-1,1), X_test7[:,7].reshape(-1,1), X_test7[:,8].reshape(-1,1), X_test6[:,9].reshape(-1,1)), axis=1)
    # val(X, y, X_test)
    # print(X.shape, y.shape, X_test.shape)
    # result = get_result_from_stacking(clf, X, y, X_test)
    # proba = clf.predict_proba(X_test)

    # 两层stacking，先36个自己stacking
    
    # X = np.concatenate((X5, X6), axis=1)
    # X_test = np.concatenate((X_test5, X_test6), axis=1)
    # result = get_result_from_stacking(clf, X, y, X_test)
    # proba = clf.predict_proba(X_test)

    # 传统与深度自己分别stacking，再加权融合
    # y_pred_proba, y, y_test_pred_proba = gen_data_for_stacking(clf, X, y, X_test, n_splits=5, random_state=RANDOM_STATE)
    # X = np.concatenate((y_pred_proba, X5), axis=1)
    # X_test = np.concatenate((y_test_pred_proba, X_test5), axis=1)
    # result = get_result_from_stacking(clf, X, y, X_test)
    # proba = clf.predict_proba(X_test)
    X = np.concatenate((X, X2, X3, X4, X5, X6, X7), axis=1)
    X_test = np.concatenate((X_test, X_test2, X_test3, X_test4, X_test5, X_test6, X_test7), axis=1)
    y_pred_proba, y, y_test_pred_prob = gen_data_for_stacking(clf, X, y, X_test, n_splits=5, random_state=RANDOM_STATE)
    val(y_pred_proba, y, y_test_pred_prob)
    
    # y_pred_proba, y, y_test_pred_prob = gen_data_for_stacking(clf, X, y, X_test, n_splits=5, random_state=RANDOM_STATE)
    y_pred_proba = y_pred_proba*0.7 + X6*0.1 + X5 * 0.1 + X7*0.1
    y_test_pred_proba =  y_test_pred_prob *0.7 + X_test6 *0.1 + X_test5 + X_test7 *0.1
    val(y_pred_proba, y, y_test_pred_prob)
    # result = get_result_from_stacking(clf, X, y, X_test)
    # proba = clf.predict_proba(X_test)

    # clf = OneVsRestClassifier(LGBMClassifier(learning_rate=0.01, boosting_type='gbdt', num_leaves=31, max_depth=7, num_class=10,
    #                       subsample=0.6, colsample_bytree=0.65, n_estimators=500, min_child_weight=9,
    #                       silent=True, reg_alpha=0.01, objective='multiclass'))

    # if save:
    #     save_url = from_project_root("processed_data/vector/{}_dc_idf_xgb.pk".format(X.shape[1] // N_CLASSES))
    #     joblib.dump(y_pred_proba, y, y_test_pred_proba, save_url)
    #     return
    
    
    test_public = pd.read_csv(test_url)
    len_list = [len(l.split(' ')) for l in test_public['word_seg']]
    #no_label = 0
    output_str = 'content_id,subject,sentiment_value,sentiment_word\n'
    for jjj in range(len(y_test_pred_proba)):
        if len_list[jjj] > 50:
            aaa = np.arange(10)
            labels = aaa[y_test_pred_proba[jjj]>0.5]
            if len(labels) > 2:
                labels = y_test_pred_proba[jjj].argsort()[-2:][::-1]
        else:
            labels = [y_test_pred_proba[jjj].argmax()]
        

        # 如果标签没有分类结果
        
        if len(labels) == 0:
            #no_label += 1
            labels = y_test_pred_proba[jjj].argmax()    # 选择有概率最高的作为分类
            # print(labels)
            # print(test_public['id'][jjj])
            #output_str += "%s,%s,0,\n" % (test_public['id'][jjj], '无')     # 留出无分类的用单标签分类模型分类
            #continue

        if type(labels) == np.int64:
            output_str += "%s,%s,0,\n" % (test_public['id'][jjj], SUBJECT_LIST[labels])
            continue
        print(labels)
        for kkk in labels:
            output_str += "%s,%s,0,\n" % (test_public['id'][jjj], SUBJECT_LIST[kkk])
    #print('%d no label' % no_label)
    outfile = open('50config.csv', 'w')
    outfile.write(output_str)
    outfile.close()
    pass




if __name__ == '__main__':
    main()
