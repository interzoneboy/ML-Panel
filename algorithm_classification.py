"""

This module wraps machine learning algorithms (scikit), calls them using a uniform interface
and data supply structure, and returns a common set of results.

"""

import os, os.path
import numpy as np
from copy import deepcopy
import pandas as pd
import pylab

from sklearn.metrics import roc_auc_score
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


def tc(func):
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception,e:
            return e
    return wrapped

def tc2(func):
    try:
        return func()
    except Exception,e:
        return e


def takeBigger(a1,a2):
    if a1 >= a2:
        return a1
    else:
        return a2



def calc_metrics(y_true, y_pred, scores=None):
    """ Calculate a standard set of performance metrics (assuming y_true and y_pred are binary targets) and return as a dict
    
    :param y_true: Ground truth labels.
    :param y_pred: Our predicted label.
    :param scores: The generated scores.
    :returns: A dictionary of performance metrics comparing y_true to y_pred.
    """
    auc = roc_auc_score(y_true, scores)

    return {'auc':auc}


def fitAndScore(model, data):
    """ Run *model* on *data*, returning a dict of results (plot function, scores, groundTruth, perf. metrics, model obj).

    :param model: A scikit learn model function (with "fit/score/predict/decision_function" methods
    :param data: Dict containing data -- {"test_x","test_y","train_x","train_y"}
    :returns: Dict with all results needed downstream.
    """
    model.fit(data['train_x'], data['train_y'])
    sc = tc2(lambda : model.decision_function(data['test_x']))
    acc = tc2(lambda : model.score(data['test_x'], data['test_y']))
    pred = tc2(lambda : model.predict(data['test_x']))
    metrics = tc2(lambda : calc_metrics(data['test_y'], pred, sc))

    return {'scores':sc,
            'acc':acc,
            'truth':data['test_y'],
            'pred':pred,
            'metrics':metrics,
            'model':model}



def raw_value(data, colIndex=None, **kwargs):
    """ Use raw value in column as classifier. """
    auc = roc_auc_score(data['test_y'], data['test_x'][:, columnIndex])
    return {"scores":data['test_x'][:, columnIndex], "truth":data['test_y'], "auc":auc, "acc":None, "model":None}


def ml_svm(data, regParam=None, **kwargs):
    """ Fit an SVM on data, regularized by regParam. """
    m = SVC(C=regParam)
    return fitAndScore(m, data)


def ml_lda(data, regParam=None, **kwargs):
    """ Fit LDA with SVD solver on data, if regParam is not supplied.
        Fit SDA with least squares solver, if regParam supplied, shrunk using Ledoit-Wolf 
        lemma (using 'auto' as shrinkage arg). """
    if regParam is None:
        m = LDA(solver='svd')
    else:
        m = LDA(solver='lsqr', shrinkage='auto')
    return fitAndScore(m, data)


def ml_qda(data, regParam=None, **kwargs):
    """ Fit QDA on data, regularized by regParam. """
    m = QDA(reg_param=regParam)
    return fitAndScore(m, data)


def ml_logistic(data, regParam=None, **kwargs):
    """ Fit logistic regression on data, shrunk using L1 penalty by regParam. """
    m = LogisticRegression(C=regParam, penalty='l1', tol=1e-6)
    return fitAndScore(m, data)


def ml_tree(data, regParam=None, **kwargs):
    """ Fit a decision tree of max height regParam to data. """
    m = DecisionTreeClassifier(max_depth=regParam)
    return fitAndScore(m, data)


def ml_sgdLog(data, regParam=None, l1_ratio=None, **kwargs):
    """ Fit Stochastic Gradient Descent with specified l1_ratio (elasticnet-ish-ness),
        regularized by regParam."""
    m = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=l1_ratio, alpha=regParam)
    return fitAndScore(m, data)


def ml_randomForest(data, regParam, numEstimators, **kwargs):
    """ Fit a Random Forest, using <numEstimators> trees of max depth <regParam>. """
    m = RandomForestClassifier(n_estimators=numEstimators, max_depth=regParam)
    return fitAndScore(m, data)



