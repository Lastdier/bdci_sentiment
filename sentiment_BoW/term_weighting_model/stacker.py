# coding: utf-8
# created by deng on 8/5/2018

from itertools import product
from collections import Counter
from os.path import exists
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.base import clone
import scipy as sp
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from utils.path_util import from_project_root, basename
from term_weighting_model.transformer import generate_vectors
from utils.data_util import load_to_df
from utils.proba_util import predict_proba
import fire


N_CLASSES = 2
RANDOM_STATE = 19950717
DROP_WORDS = 0
N_JOBS = 10
CV = 5


def load_params():
    """ load params

    Returns:
        list, list of params dict

    """
    # Hybrid dc idf
    params_grad = [
        {
            'column': ['word_seg'],
            'trans_type': ['dc', 'idf'],
            'max_n': [1],
            'min_df': [2],
            'max_df': [0.9],
            'max_features': [20000],
            'balanced': [False, True],
            're_weight': [0]
        },  # 4
        {
            'column': ['word_seg', 'article'],
            'trans_type': ['dc'],
            'max_n': [2],
            'min_df': [2],
            'max_df': [0.8],
            'max_features': [40000, 5000],
            'balanced': [False, True],
            're_weight': [9]
        },  # 8
        {
            'column': ['word_seg', 'article'],
            'trans_type': ['dc'],
            'max_n': [3],
            'min_df': [2],
            'max_df': [0.8],
            'max_features': [40000, 5000],
            'balanced': [False, True],
            're_weight': [0, 9]
        },  # 16
        {
            'column': ['word_seg', 'article'],
            'trans_type': ['idf'],
            'max_n': [3],
            'min_df': [2],
            'max_df': [0.8],
            'max_features': [40000, 5000],
            'balanced': [False, True],
        },  # 8
    ]  # 36


    params_list = list()
    for params_dict in params_grad:
        keys, value_lists = zip(*(params_dict.items()))
        for prod in product(*value_lists):
            params_list.append(dict(zip(keys, prod)))

    return params_list


def run_parallel(index, train_url, test_url, params, clf, n_splits, random_state,
                 use_proba=False, verbose=False, drop_words=0):
    """ for run cvs parallel

    Args:
        index: index to know which cv it belongs to
        train_url: train data url
        test_url: teat data url
        params: params for generate_vectors
        clf: classifier
        n_splits: n_splits for KFold
        random_state: random_state for KFold
        use_proba: True to predict probabilities of labels instead of labels
        verbose: True to print more info
        drop_words: drop_words for generate_vectors

    Returns:

    """

    X, y, X_test = generate_vectors(train_url, test_url, drop_words=drop_words, verbose=verbose, **params)
    if not sp.sparse.isspmatrix_csr(X):
        X = sp.sparse.csr_matrix(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=bool(random_state), random_state=random_state)
    y_pred = np.zeros((X.shape[0], 1))
    y_pred_proba = np.zeros((X.shape[0], N_CLASSES))
    y_test_pred_proba = np.zeros((X_test.shape[0], N_CLASSES))
    for ind, (train_index, cv_index) in enumerate(skf.split(X, y)):
        X_train, X_cv = X[train_index], X[cv_index]
        y_train, y_cv = y[train_index], y[cv_index]
        clf.fit(X_train, y_train)
        y_pred[cv_index] = clf.predict(X_cv).reshape(-1, 1)
        y_pred_proba[cv_index] = clf._predict_proba_lr(X_cv)
        print("%d/%d cv macro f1 of params set #%d:" % (ind + 1, n_splits, index),
              f1_score(y_cv, y_pred[cv_index], average='micro'))
        y_test_pred_proba += clf._predict_proba_lr(X_test)
    print("#%d macro f1: " % index, f1_score(y, y_pred, average='micro'))

    y_test_pred = clf.predict(X_test).reshape(X_test.shape[0], 1)
    y_test_pred_proba /= n_splits  # normalize to 1
    if not use_proba:
        return index, y_pred, y_test_pred
    return index, y_pred_proba, y_test_pred_proba


def feature_stacking(n_splits=CV, random_state=None, use_proba=False, verbose=False, drop_words=DROP_WORDS, name=None):
    """

    Args:
        n_splits: n_splits for KFold
        random_state: random_state for KFlod
        use_proba: True to predict probabilities of labels instead of labels
        verbose: True to print more info
        drop_words: drop_words for run_parallel

    Returns:
        X, y, X_test

    """

    clf = LinearSVC()
    train_url = from_project_root("data/multilabel_"+name+".csv")
    test_url = from_project_root("data/test_processed.csv")
    # test_url = None
    X, y, X_test = generate_vectors(train_url, test_url, sublinear_tf=False)  # for X.shape
    
    params_list = load_params()
    parallel = joblib.Parallel(n_jobs=N_JOBS, verbose=True)
    rets = parallel(joblib.delayed(run_parallel)(
        ind, train_url, test_url, params, clone(clf), n_splits, random_state, use_proba, verbose, drop_words
    ) for ind, params in enumerate(params_list))
    rets = sorted(rets, key=lambda x: x[0])

    X_stack_train = np.empty((X.shape[0], 0), float)
    X_stack_test = np.empty((X_test.shape[0], 0), float)
    for ind, y_pred, y_pred_test in rets:
        X_stack_train = np.append(X_stack_train, y_pred, axis=1)
        X_stack_test = np.append(X_stack_test, y_pred_test, axis=1)

    return X_stack_train, y, X_stack_test


def model_stacking_from_pk(model_urls):
    """

    Args:
        model_urls: model stacking from model urls

    Returns:
        X, y, X_test: stacked new feature

    """
    if model_urls is None or len(model_urls) < 1:
        print("invalid model_urls")
        return

    print("files for stacking ...")
    for url in model_urls:
        print(' ', basename(url))

    X, y, X_test = joblib.load(model_urls[0])
    for url in model_urls[1:]:
        X_a, _, X_test_a = joblib.load(url)
        X = np.append(X, X_a, axis=1)
        X_test = np.append(X_test, X_test_a, axis=1)

    return X, y, X_test


def gen_data_for_stacking(clf, X, y, X_test, n_splits=5, random_state=None,name=None):
    """ generate single model result data for stacking

    Args:
        clf: single model
        X: original X
        y: original y
        X_test: original X_test
        n_splits: n_splits for skf
        random_state: random_state for skf

    Returns:
        X, y, X_test

    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=bool(random_state), random_state=random_state)
    y_pred = np.zeros((X.shape[0],))  # for printing score of each fold
    y_pred_proba = np.zeros((X.shape[0], N_CLASSES))
    y_test_pred_proba = np.zeros((X_test.shape[0], N_CLASSES))
    for ind, (train_index, cv_index) in enumerate(skf.split(X, y)):  # cv split
        X_train, X_cv = X[train_index], X[cv_index]
        y_train, y_cv = y[train_index], y[cv_index]
        clf.fit(X_train, y_train)
        y_pred[cv_index] = clf.predict(X_cv)
        y_pred_proba[cv_index] = predict_proba(clf, X_cv)
        print("%d/%d cv macro f1 :" % (ind + 1, n_splits),
              f1_score(y_cv, y_pred[cv_index], average='micro'))
        y_test_pred_proba += predict_proba(clf, X_test)
    print("micro f1:", f1_score(y, y_pred, average='micro'))  # calc macro_f1 score
    joblib.dump(y_pred, 'y_pred_'+name+'.pk')

    y_test_pred_proba /= n_splits  # normalize to 1
    return y_pred_proba, y, y_test_pred_proba


def generate_meta_feature(data_url, normalize=True):
    """ generate meta feature

    Args:
        data_url: url to data
        normalize: normalize result into [0, 1]

    Returns:
        generated meta DataFrame

    """
    save_url = data_url.replace('.csv', '_meta_df.pk')
    if exists(save_url):
        return joblib.load(save_url)

    data_df = load_to_df(data_url)
    meta_df = pd.DataFrame()

    for level in ('word_seg', 'article'):
        # word num
        meta_df[level + '_num'] = data_df[level].apply(lambda x: len(x.split()))
        # different word num
        meta_df[level + '_unique'] = data_df[level].apply(lambda x: len(set(x.split())))
        # most common word num
        meta_df[[level + '_common', level + '_common_num']] = pd.DataFrame(data_df[level].apply(
            lambda x: Counter(x.split()).most_common(1)[0]).tolist()).astype(int)

    # average phrase len
    meta_df['avg_phrase_len'] = meta_df['article_num'] / meta_df['word_seg_num']

    # normalization
    if normalize:
        for col in meta_df:
            meta_df[col] -= meta_df[col].min()
            meta_df[col] /= meta_df[col].max()

    joblib.dump(meta_df, save_url)
    return meta_df


def main(name):
    params = load_params()
    print("len(params) =", len(params))
    save_url = from_project_root("processed_data/vector/stacked_dc_idf_"+name+"_%d.pk" % len(load_params()))
    joblib.dump(feature_stacking(use_proba=True, random_state=RANDOM_STATE, name=name), save_url)


if __name__ == '__main__':
    fire.Fire()