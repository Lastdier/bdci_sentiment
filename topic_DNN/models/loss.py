import torch as t
def multilabelloss():
    return  t.nn.MultiLabelSoftMarginLoss()
def multilabel_marginloss():
    return t.nn.MultiLabelMarginLoss()
def bceloss():
    return t.nn.BCELoss()
def bcewithlogitloss():
    return t.nn.BCEWithLogitsLoss()