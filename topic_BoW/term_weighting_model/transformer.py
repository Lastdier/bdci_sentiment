# coding: utf-8
# created by deng on 8/2/2018

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import defaultdict
from sklearn.feature_extraction.text import _document_frequency
from time import time

import numpy as np
import scipy.sparse as sp
import sys
sys.path.append("..")
from utils.path_util import from_project_root
from utils.data_util import load_to_df
import pandas as pd


class TfdcTransformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a normalized tf or tf-dc representation
    Tf means term-frequency while tf-dc means distributional-concentration
    times term-frequency.

    Tf is "n" (natural) by default, "l" (logarithmic) when
    ``sublinear_tf=True``.

    Parameters
    ----------
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    use_dc : boolean, default=True
        Enable dc reweighting.
    smooth_dc : boolean, default=True
        Smooth dc weights by adding one to f(t,Ci) and f(t). Prevents
        zero divisions and zero log.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    balanced: boolean, default=False
        use bdc instead of dc
    re_weight: int or float, default=0
        if not zero, use 1 + re_weight*dc instead of dc

    References
    ----------
    .. [2015 ICTAI] `Tao Wang, Yi Cai. Entropy-based TermWeighting Schemes for
                     Text Categorization in VSM`
    """

    def __init__(self, norm='l2', use_dc=True, smooth_dc=True, sublinear_tf=False,
                 balanced=False, re_weight=0):
        self.norm = norm
        self.use_dc = use_dc
        self.smooth_dc = smooth_dc
        self.sublinear_tf = sublinear_tf
        self.balanced = balanced
        self.re_weight = re_weight

    def fit(self, X, y):
        """Learn the idf vector (global term weights)
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        y : labels of each samples, [n_samples,]
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        y = np.asarray(y)  # pd.Series don't support csr slicing

        if self.use_dc:

            n_samples, n_features = X.shape

            # calculate term frequencies, f(t)
            tfs = np.asarray(X.sum(axis=0)).ravel()
            tps = np.zeros(n_features)  # sum of p(t, Ci) for bdc

            # initialize dc
            dc = np.ones(n_features)
            smooth_dc = int(self.smooth_dc)

            # all labels
            # labels = set(y)
            # print(X.shape, y.shape)
            for label in range(10):
                iii_X = []  #存储有当前标签label的样本下标；
                for k in range(len(y)):
                    if y[k][label] == 1:
                        iii_X.append(k)
                label_X = X[iii_X]
                label_tfs = np.asarray(label_X.sum(axis=0)).ravel()  # f(t, Ci)

                # calculate sum of p(t, Ci) for bdc
                if self.balanced:
                    tps = tps + label_tfs / label_tfs.sum()  # p(t, Ci) for bdc

                # label_tfs might have 0 values in some columns, need to smooth
                label_h = label_tfs / tfs  # f(t, Ci) / f(t)
                label_h[label_h == 0] = smooth_dc  # to avoid zero log
                dc += label_h * np.log(label_h) / np.log(10)

            # calculate balanced dc
            if self.balanced:
                # initialize bdc
                dc = np.ones(n_features)
                for label in range(10):
                    iii_X = []
                    for k in range(len(y)):
                        if y[k][label] == 1:
                            iii_X.append(k)
                    label_X = X[iii_X]
                    label_tfs = np.asarray(label_X.sum(axis=0)).ravel()  # f(t, Ci) for dc
                    label_tps = label_tfs / label_tfs.sum()  # p(t, Ci) for bdc
                    label_h = label_tps / tps
                    label_h[label_h == 0] = smooth_dc  # to avoid zero log
                    dc = dc + label_h * np.log(label_h) / np.log(10)

            if self.re_weight > 0:
                dc = 1 + dc * self.re_weight

            self._dc_diag = sp.spdiags(dc, diags=0, m=n_features, n=n_features, format='csr')
        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_dc:
            check_is_fitted(self, '_dc_diag', 'dc vector is not fitted')

            expected_n_features = self._dc_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._dc_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    @property
    def dc_(self):
        # if _dc_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "dc_") is False
        return np.ravel(self._dc_diag.sum(axis=0))


def generate_vectors(train_url, test_url=None, column='article', trans_type=None, max_n=1, min_df=1, max_df=1.0,
                     max_features=1, sublinear_tf=True, balanced=False, re_weight=0, verbose=False, drop_words=0):
    """ generate X, y, X_test vectors with csv(with header) url use pandas and CountVectorizer

    Args:
        train_url: url to train csv
        test_url: url to test csv, set to None if not need X_test
        column: column to use as feature
        trans_type: specific transformer, {'dc','idf'}
        max_n: max_n for ngram_range
        min_df: min_df for CountVectorizer
        max_df: max_df for CountVectorizer
        max_features: max_features for CountVectorizer
        sublinear_tf: sublinear_tf for default TfdcTransformer
        balanced: balanced for default TfdcTransformer, for idf transformer, it is use_idf
        re_weight: re_weight for TfdcTransformer
        verbose: True to show more information
        drop_words: randomly delete some words from sentences

    Returns:
        X, y, X_test

    """
    #verbose and print("loading '%s' level data from %s with pandas" % (column, train_url))

    train_df = load_to_df(train_url)

    # vectorizer
    vec = CountVectorizer(ngram_range=(1, max_n), min_df=min_df, max_df=max_df,
                          max_features=max_features, token_pattern='\w+')
    s_time = time()
    #verbose and print("finish loading, vectorizing")
    #verbose and print("vectorizer params:", vec.get_params())

    sequences = train_df[column]
    # delete some words randomly *********************
    # for i, row in enumerate(sequences):
    #     if drop_words <= 0:
    #         break
    #     if np.random.ranf() < drop_words:
    #         row = np.array(row.split())
    #         sequences.at[i] = ' '.join(row[np.random.ranf(row.shape) > 0.35])

    X = vec.fit_transform(sequences)
    e_time = time()
    #verbose and print("finish vectorizing in %.3f seconds, transforming" % (e_time - s_time))
    # transformer
    if trans_type is None or trans_type == 'idf':
        trans = TfidfTransformer(sublinear_tf=sublinear_tf, use_idf=balanced)
    else:
        trans = TfdcTransformer(sublinear_tf=sublinear_tf, balanced=balanced, re_weight=re_weight)

    #verbose and print("transformer params:", trans.get_params())

    #binarilize label
    labels = []
    for i in train_df["class"]:
        temp = [0] * 10
        for j in i.split(' '):
            temp[int(j)] = 1
        labels.append(temp)
    y = np.array(labels)
    X = trans.fit_transform(X, y)
    X_test = None
    if test_url:
        #verbose and print("transforming test set")
        test_df = load_to_df(test_url)
        X_test = vec.transform(test_df[column])
        X_test = trans.transform(X_test)

    s_time = time()
    #verbose and print("finish transforming in %.3f seconds\n" % (s_time - e_time))
    return X, y, X_test


class Vector_generator(BaseEstimator, TransformerMixin):
    def __init__(self, column='article', trans_type=None, max_n=1, min_df=1, max_df=1.0,
                        max_features=1, sublinear_tf=True, balanced=False, re_weight=0):
        

        # vectorizer
        if column == 'article':
            self.col = 0
        elif column == 'word_seg':
            self.col = 1
        print(trans_type)
        self.vec = CountVectorizer(ngram_range=(1, max_n), min_df=min_df, max_df=max_df,
                            max_features=max_features, token_pattern='\w+')
        
        if trans_type is None or trans_type == 'idf':
            self.trans = TfidfTransformer(sublinear_tf=sublinear_tf, use_idf=balanced)
        else:
            self.trans = TfdcTransformer(sublinear_tf=sublinear_tf, balanced=balanced, re_weight=re_weight)

        # sequences = train_df[column]


    def fit(self, X, y):
        X = X[:, self.col]
        X = vec.fit_transform(X)
        # transformer



        #binarilize label
        labels = []
        for i in y:
            temp = [0] * 10
            for j in i.split(' '):
                temp[int(j)] = 1
            labels.append(temp)
        _y = np.array(labels)
        X = trans.fit_transform(X, _y)

        return self
    
    def transform(self, X):
        return X


def main():
    train_url = from_project_root("data/multilabel.csv")
    test_url = from_project_root("data/test_processed.csv")
    generate_vectors(train_url, test_url, 'word_seg')
    pass


if __name__ == '__main__':
    main()
