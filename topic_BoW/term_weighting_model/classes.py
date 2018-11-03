# coding: utf-8
# created by deng on 2018/11/3
from sklearn.svm import LinearSVC


class LinearSVCP(LinearSVC):
    def predict_proba(self, X):
        return self._predict_proba_lr(X)


def main():
    pass


if __name__ == '__main__':
    main()
