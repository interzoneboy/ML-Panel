"""

This module wraps machine learning algorithms (scikit), calls them using a uniform interface
and data supply structure, and returns a common set of results.

"""

import os, os.path
import numpy as np
from copy import deepcopy
import pandas as pd
import pylab
import traceback

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

def tc2(func, context=None):
    try:
        return func()
    except Exception,e:
        if context is None:
            return e
        else:
            e.context = context
            return e


def takeBigger(a1,a2):
    if a1 >= a2:
        return a1
    else:
        return a2


# Module object for holding prepared algorithm objects.
algorithms = {}


# Algorithm function decorators

def reg(arg, whichList=None):
    def decor(func):
        algorithms[arg] = func
        if whichList!=None:
            whichList[arg] = func

        return func
    return decor

def opt(paramName, valsList):
    def decor(func):
        # Have we passed in a parameter
        def wrapped(*args, **kwargs):
            #if len(args)!=0:
            if paramName in kwargs.keys():
                #dataToUse = getData() if 'data' not in kwargs.keys() else kwargs['data']
                #kwargs['data'] = dataToUse
                return func(*args, **kwargs)
            else:
                #dataToUse = getData() if 'data' not in kwargs.keys() else kwargs['data'] #kwargs['data']
                acc = []
                x_vals = []; y_vals = [];
                for val in list(valsList):
                    kwargs[paramName] = val
                    #kwargs['data'] = dataToUse
                    output = func(*args, **kwargs)
                    output['best_'+paramName] = val
                    x_vals.append(val)
                    try:
                        y_vals.append(output['metrics']['auc'])
                    except Exception, e:
                        tb = traceback.format_exc()
                        from IPython.terminal.embed import InteractiveShellEmbed
                        ipshell = InteractiveShellEmbed(banner1="opt breakpoint")
                        ipshell()
                        raise

                    #print output['metrics']['auc']
                    acc.append(output)
                accSrt = sorted(acc, key=lambda x: x['metrics']['auc'], reverse=True)
                to_return = accSrt[0]
                to_return['opt_output'] = {'metric':'auc', 'series':{'x':x_vals, 'y':y_vals}}
                return(to_return)
        wrapped.func_name = func.func_name
        return wrapped
    return decor


####################################################33333
######################################################333
########################################################3

# Classification Algorithm Boilerplate


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
    try:
        model.fit(data['train_x'], data['train_y'])
        sc = tc2(lambda : model.decision_function(data['test_x']))
        acc = tc2(lambda : model.score(data['test_x'], data['test_y']))
        pred = tc2(lambda : model.predict(data['test_x']))
        metrics = tc2(lambda : calc_metrics(data['test_y'], pred, sc), {'truth':data['test_y'], 'pred':pred, 'scores':sc})
    except Exception, e:
        tb = traceback.format_exc()
        from IPython.terminal.embed import InteractiveShellEmbed
        ipshell = InteractiveShellEmbed(banner1="fitAndScore breakpoint")
        ipshell()
        raise


    return {'scores':sc,
            'acc':acc,
            'truth':data['test_y'],
            'pred':pred,
            'metrics':metrics,
            'model':model}


####################################################33333
######################################################333
########################################################3

# Algorithms




def raw_value(data, colIndex=None, **kwargs):
    """ Use raw value in column as classifier. """
    auc = roc_auc_score(data['test_y'], data['test_x'][:, columnIndex])
    return {"scores":data['test_x'][:, columnIndex], "truth":data['test_y'], "auc":auc, "acc":None, "model":None}

@reg("svm_rbf")
@opt("regParam", np.logspace(-15, 15, num=50))
def ml_svm_rbf(data, regParam=None, **kwargs):
    """ Fit an SVM on data, regularized by regParam. """
    m = SVC(C=regParam, kernel='rbf')
    return fitAndScore(m, data)

@reg("svm_linear")
@opt("regParam", np.logspace(-15, 15, num=50))
def ml_svm_linear(data, regParam=None, **kwargs):
    """ Fit an SVM on data, regularized by regParam. """
    m = SVC(C=regParam, kernel='linear')
    return fitAndScore(m, data)

@reg("lda")
def ml_lda(data, regParam=None, **kwargs):
    """ Fit LDA with SVD solver on data, if regParam is not supplied.
        Fit SDA with least squares solver, if regParam supplied, shrunk using Ledoit-Wolf 
        lemma (using 'auto' as shrinkage arg). """
    if regParam is None:
        m = LDA(solver='svd')
    else:
        m = LDA(solver='lsqr', shrinkage='auto')
    return fitAndScore(m, data)

@reg("qda")
@opt("regParam", np.linspace(-5, 10, num=50))
def ml_qda(data, regParam=None, **kwargs):
    """ Fit QDA on data, regularized by regParam. """
    m = QDA(reg_param=regParam)
    return fitAndScore(m, data)

@reg("logistic")
@opt("regParam", np.logspace(-15, 15, num=50))
def ml_logistic(data, regParam=None, **kwargs):
    """ Fit logistic regression on data, shrunk using L1 penalty by regParam. """
    m = LogisticRegression(C=regParam, penalty='l1', tol=1e-6)
    return fitAndScore(m, data)

@reg("tree")
@opt("regParam", range(1,11))
def ml_tree(data, regParam=None, **kwargs):
    """ Fit a decision tree of max height regParam to data. """
    m = DecisionTreeClassifier(max_depth=regParam)
    return fitAndScore(m, data)

@reg("SGDlogistic")
@opt("regParam", np.logspace(-15, 15, num=50))
def ml_sgdLog(data, regParam=None, l1_ratio=1.0, **kwargs):
    """ Fit Stochastic Gradient Descent with specified l1_ratio (elasticnet-ish-ness),
        regularized by regParam."""
    m = SGDClassifier(loss='log', penalty='elasticnet', l1_ratio=l1_ratio, alpha=regParam)
    return fitAndScore(m, data)

@reg("forest")
@opt("regParam", range(1,11))
def ml_randomForest(data, regParam, numEstimators=10, **kwargs):
    """ Fit a Random Forest, using <numEstimators> trees of max depth <regParam>. """
    m = RandomForestClassifier(n_estimators=numEstimators, max_depth=regParam)
    return fitAndScore(m, data)




