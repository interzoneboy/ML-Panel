"""

"""

import os, os.path
import sys
import copy
import warnings
import numpy as np
import inspect
import pylab
import traceback
from contextlib import contextmanager

from datasets2 import getRandomSplit, df_cont
import datasets2


def getDataFunction(df=df_cont, mriVar="OMRI_pct_viscfat_L4L5", genderKey="both"):
    stuff = getRandomSplit(df, mriVar, genderKey)
    return (stuff['data_tr'], stuff['labels_tr'], stuff['data_test'], stuff['labels_test'])
getData = getDataFunction


def copyData(dataIn):
    return( (copy.deepcopy(dataIn[0]),
             copy.deepcopy(dataIn[1]),
             copy.deepcopy(dataIn[2]),
             copy.deepcopy(dataIn[3]))
          )

@contextmanager
def stdout_redirected(new_stdout):
    # This maps both stdout and stderr to the new_stdout file descriptor.
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = new_stdout
    sys.stderr = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr



from copy import deepcopy
from IPython import embed

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import RANSACRegressor

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

all_functions = {}
singleVarOpt_functions = {}
optimizedFunctions = {}


def opt(paramName, metricName, valsList):
    def decor(func):
        # Have we passed in a parameter
        def wrapped(*args, **kwargs):
            #if len(args)!=0:
            if paramName in kwargs.keys():
                dataToUse = getData() if 'data' not in kwargs.keys() else kwargs['data']
                kwargs['data'] = dataToUse
                return func(*args, **kwargs)
            else:
                dataToUse = getData() if 'data' not in kwargs.keys() else kwargs['data'] #kwargs['data']
                acc = []
                for val in list(valsList):
                    kwargs[paramName] = val
                    kwargs['data'] = dataToUse
                    output = func(*args, **kwargs)
                    output['best_'+paramName] = val
                    #print output['auc']
                    acc.append(output)
                accSrt = sorted(acc, key=lambda x: x[metricName], reverse=True)
                return(accSrt[0])
        wrapped.func_name = func.func_name
        return wrapped
    return decor

def reg(arg, whichList=None):
    def decor(func):
        all_functions[arg] = func
        if whichList!=None:
            whichList[arg] = func

        return func
    return decor

def regOptimized(arg):
    def decor(func):
        optimizedFunctions[arg] = func
        return func
    return decor


def calc_metrics(y_true, y_pred):
    ev = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    #medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true,y_pred)
    return {'explainedVariance':ev,
            'mean_absolute_err':mae,
            'mean_squared_err':mse,
            #'median_absolute_err':medae,
            'r2_coeff_det':r2}

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


def fitAndScore(model, dataTuple):
    d_tr, lab_tr, d_test, lab_test = dataTuple
    model.fit(d_tr, lab_tr)
    acc =  tc2(lambda : model.score(d_test, lab_test))
    sc =  tc2(lambda : model.decision_function(d_test))
    pred = tc2(lambda : model.predict(d_test))
    metrics = tc2(lambda : calc_metrics(lab_test, pred))
    def plot():
        pylab.scatter(lab_test, pred)
        pylab.plot([min(lab_tr), max(lab_tr)],[min(lab_tr),max(lab_tr)],'k--', lw=3)
        pylab.show()
    rd = {'scores':sc, 
            'acc':acc, 
            'truth':lab_test, 
            'pred': pred, 
            'metrics':metrics,
            'model':model,
            'plot':plot}
    return rd


@regOptimized("linear")
@reg("linear")
def ml_linear(data=None, **kwargs):
    
    model = LinearRegression()
    return fitAndScore(model, data)

@regOptimized("ransac_linear")
@reg("ransac_linear")
def ml_ransacLinear(data=None, **kwargs):

    model = RANSACRegressor(LinearRegression())
    return fitAndScore(model, data)

@reg("sgd", singleVarOpt_functions)
@opt("regParam", "acc", np.logspace(-8,5,num=50))
def ml_sgd(data=None, regParam=0.0001, **kwargs):

    s = SGDRegressor(loss="squared_loss", penalty='l1', alpha=regParam)
    return fitAndScore(s, data)

@regOptimized("sgd")
def opt_sgd(data=None, **kwargs):
    return ml_sgd(data=data, regParam=0.00000001, **kwargs)

@reg("lasso", singleVarOpt_functions)
@opt("regParam", "acc", np.logspace(-8,5,num=50))
def ml_lasso(data=None, regParam=0.1, **kwargs):
    model = Lasso(alpha=regParam)
    return fitAndScore(model, data)

@regOptimized("lasso")
def opt_lasso(data=None, **kwargs):
    return ml_lasso(data=data, regParam=0.015, **kwargs)

@reg("ransac_lasso")
#@opt("regParam", "acc", np.logspace(-8,5,num=50))
def ml_ransacLasso(data=None, regParam=0.05, **kwargs):
    model = RANSACRegressor(Lasso(alpha=regParam))
    return fitAndScore(model,data)


@reg("lassoCV")
@regOptimized("lassoCV")
def ml_lassoCV(data=None, **kwargs):
    model = LassoCV()
    return fitAndScore(model, data)
    

@reg("lassoLarsCV")
@regOptimized("lassoLarsCV")
def ml_lassoLarsCV(data=None, **kwargs):
    model = LassoLarsCV()
    return fitAndScore(model, data)

@reg("ridge", singleVarOpt_functions)
@opt("regParam", "acc", np.logspace(-8, 5, num=50))
def ml_ridge(data=None, regParam=0.01, **kwargs):
    model = Ridge(alpha=regParam)
    return fitAndScore(model, data)

@regOptimized("ridge")
def opt_ridge(data=None, **kwargs):
    return ml_ridge(data=data, regParam=0.75, **kwargs)

@reg("ridgeCV")
@regOptimized("ridgeCV")
def ml_ridgeCV(data=None, **kwargs):
    model = RidgeCV()
    return fitAndScore(model, data)



