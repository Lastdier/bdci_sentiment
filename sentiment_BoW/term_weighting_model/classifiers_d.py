# coding: utf-8
# created by deng on 7/27/2018

from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import fire
from sklearn.metrics import f1_score, accuracy_score
from time import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
import sys
sys.path.append("..")
from utils.path_util import from_project_root
from utils.proba_util import predict_proba
from term_weighting_model.transformer import generate_vectors
from term_weighting_model.stacker import generate_meta_feature, gen_data_for_stacking
from term_weighting_model.stacker import model_stacking_from_pk

N_JOBS = 1
N_CLASSES = 2
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
        for train_index, cv_index in slf.split(X, y):
            X_train = X[train_index]
            y_train = y[train_index]
            clf.fit(X_train, y_train)
            y_pred_proba += predict_proba(clf, X_test, X_train, y_train)
        y_pred_proba /= n_splits
        y_pred = y_pred_proba.argmax(axis=1) + 1

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


def main(name):
    # load data from pickle
    pk_url = from_project_root("processed_data/vector/stacked_dc_idf_"+name+"_36.pk")
    print("loading data from", pk_url)
    X, y, X_test = joblib.load(pk_url)

    train_url = from_project_root("data/multilabel_"+name+".csv")
    test_url = from_project_root("data/test_processed.csv")

    print(X.shape, y.shape, X_test.shape)
    clf = XGBClassifier(n_jobs=-1)  # xgb's default n_jobs=1

    result = get_result_from_stacking(clf, X, y, X_test)
    test_public = pd.read_csv(test_url)['id']
    output_str = 'content_id,subject,sentiment_value,sentiment_word\n'
    for jjj in range(len(result)):
        output_str += "%s,0,%s,\n" % (test_public[jjj], result[jjj])
    outfile = open('result_36'+name+'.csv', 'w')
    outfile.write(output_str)
    outfile.close()

    save_url = from_project_root("processed_data/vector/{}_dc_idf_xgb.pk".format(X.shape[1] // N_CLASSES))
    joblib.dump(gen_data_for_stacking(clf, X, y, X_test, n_splits=5, random_state=19950717, name=name), save_url)

    pass


if __name__ == '__main__':
    fire.Fire()
