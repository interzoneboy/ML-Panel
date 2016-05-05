"""

Functions to handle the flow of analysis, from filename to batch 
of plots and stats.

"""

from IPython import embed
import os, os.path
import sys
import traceback
import copy
import numpy as np
import pylab
import pickle
import math
from contextlib import contextmanager

from jinja2 import Environment, FileSystemLoader


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


import dataset as DS
import algorithm_classification as ALG


def plotOpts(listOfFitAndScore):
    """
    Show some kind of histogram of optimal regularization parameters.
    """
    d = listOfFitAndScore
    bins = d[0]['opt_output']['series']['x']
    params = [a['best_regParam'] for a in d]
    inds = [a['bestIndex_regParam'] for a in d]
    pylab.hist(inds, bins=len(bins))
    locs,labels = pylab.xticks()
    print locs
    print labels
    #assert len(labels)==len(bins), "%s and %s "%(str(len(labels)), str(len(bins)))
    pylab.xticks(locs, ["{:.3e}".format(bins[int(aa)]) for aa in locs], rotation=270)
    pylab.show()

#################
#################
#################


def plotCoefBoxplot(alg_res, keyList, figPath="figures/testFig3.png"):
    """ 
    Plot the coefficients of the models. Note that only some model objects have a "coef_" attribute.
    Where the model doesn't have one there will just be an ugly blank plot.
    """
    ncol = len(keyList) / 4 + 1
    nrow = int(math.ceil(float(len(keyList)) / float(ncol)))
    pylab.figure()
    for i,ll in enumerate(keyList):
        pylab.subplot(nrow, ncol, (i+1))
        try:
            mat = np.vstack([np.array(alg_res[ll][j]['model'].coef_[0,:]) for j in range(0, len(alg_res[ll]))])
            pylab.boxplot([mat[:,i] for i in range(0, mat.shape[1])])
        except Exception,e:
            pass
        pylab.title(ll)
    #pylab.show()
    pylab.savefig(figPath)


def plotCoefHeatmap(alg_res, keyList, figPath="figures/testFig2.png"):
    """
    Plots the coeffs as a heatmap, with each model having a row. The scaled median coeffs are shown
    """
    mList = []
    names = []
    for i,ll in enumerate(keyList):
        try:
            stacko = np.vstack([np.array(alg_res[ll][j]['model'].coef_[0,:]) for j in range(0, len(alg_res[ll]))])
            stacko_meds = [np.median(stacko[:,zz]) for zz in range(0, stacko.shape[1])]
            mList.append(stacko_meds)
            names.append(ll)
        except Exception,e:
            pass

    heatData = np.vstack(mList)
    for i in range(0, heatData.shape[0]):
        fact = np.max([abs(a) for a in heatData[i,:]])
        heatData[i,:] = heatData[i,:] /  float(fact)
    
    pylab.figure()
    pylab.pcolormesh(heatData, cmap='coolwarm')
    pylab.xlabel("Feature Index")
    pylab.yticks(range(0, len(names)), names)
    #pylab.show()
    pylab.savefig(figPath)

    #return mList




def plotMetrics(alg_res, keyList, metric, figPath="figures/testFig1.png"):
    """ Plot histograms of the metric, for each classifier """
    ncol = len(keyList) / 4 + 1
    nrow = int(math.ceil(float(len(keyList)) / float(ncol)))
    pylab.figure()
    #print "%s, %s" % (str(ncol), str(nrow))
    ax_arr = []
    for i,ll in enumerate(keyList):
        if i==0:
            ax1 = pylab.subplot(nrow, ncol, (i+1))
            ax_arr.append(ax1)
        else:
            ax2 = pylab.subplot(nrow, ncol, (i+1), sharex=ax1)
            ax_arr.append(ax2)

        pylab.hist([alg_res[ll][j]['metrics'][metric] for j in range(0, len(alg_res[ll]))])
        pylab.title(ll)
        pylab.xticks(rotation=45)

    ts = ax1.get_xticks()
    #tsl = ax1.get_xticklabels()

    ignore = [a.set_xticks([ts[0], 1.0]) for a in ax_arr]

    #ax1.set_xticklabels([tsl[0], tsl[-1]])
    
    pylab.savefig(figPath)
    #return ax_arr

def test2(res):
    return plotMetrics(res, res.keys(), 'auc')

#################
#################
#################





def getPickle():

    with open("alg_results.saved", 'r') as f:
        output = pickle.load(f)
    return output

def savePickle(p):

    with open("alg_results.saved", 'w') as f:
        pickle.dump(p, f)


def loadData(dataFrameName="SampleData/example_data.csv"):
#if __name__ == "__main__":

    #res = run_test()
    d = DS.read_data_csv(dataFrameName)
    #d2 = DS.compute_quantiles(d, "y_var_3", 4)
    #d3 = DS.top_or_bottom(d2, "qtile_4_y_var_3", collapse=True)

    # Lets write out d3 for quick diagnostic plotting in R
    #d3.to_csv("debugOut.csv")
    #with open("debugOut.csv", 'w') as f:
    return d

def runMain(d, yVarName, numQuantiles):

    d = DS.compute_quantiles(d, yVarName, numQuantiles)
    name2 = "qtile_" + str(numQuantiles) + "_" + yVarName
    d = DS.top_or_bottom(d, name2, collapse=True)

    x_names = [a for a in d.columns if ('x_var' in a.lower() and ('agg' not in a.lower()))]
    y_name = "TB"+name2  #yVarName#"TBqtile_4_y_var_3"
     
    alg_results = {}
    for (nn,kk,rr) in [(None, 'svm_linear',0.008), (None, 'svm_rbf',0.5), (None, 'lda',None), (None, 'qda',0.2), 
                       (None, 'logistic',0.01), #('sgd1','SGDlogistic',0.0000001),
                       #('logisticReg', 'logistic', None),
                       ('sgd1', 'SGDlogistic', 0.01),
                       #('sgd2', 'SGDlogistic', 0.1),
                       #('sgd3', 'SGDlogistic', 0.2),
                       ('sgd4', 'SGDlogistic', 0.3),
                       #('sgd5', 'SGDlogistic', 0.4),
                       #('sgd6', 'SGDlogistic', 0.42),
                       #('sgd7', 'SGDlogistic', 12.0),
                       (None,'logisticCV', None)]:
        if nn is None:
            nn = kk
        print "Working on " + str(nn)
        alg_results[nn] = []
        for numIter in range(0, 100):
            dReady = DS.randomSplit(d, y_name, x_names)
            alg_results[nn].append(ALG.algorithms[kk](dReady, regParam=rr))
            #alg_results[kk].append(ALG.algorithms[kk](dReady))
            if numIter%20==0:
                print "Finished %s iterations." % (str(numIter),)

    return (alg_results, x_names, y_name, d)




def renderHtml(metric="auc", limit=3):

    theLoader = FileSystemLoader('templates')
    env = Environment(loader=theLoader)

    d = loadData()

    ys = [a for a in d.columns if 'y_var' in a.lower()]

    if limit is None:
        limit = len(ys)

    dataList = []

    for yInd,y in enumerate(ys[:limit]):

        print str((yInd,y))
        
        try:
            res,x,y,ignore = runMain(d, y, 4)
        except Exception,e:
            raise

        figPath = lambda x: "figures/plot_%s_%s.png" % (str(yInd), str(x))

        plotCoefBoxplot(res, res.keys(), figPath=figPath(1))
        plotCoefHeatmap(res, res.keys(), figPath=figPath(2))
        plotMetrics(res, res.keys(), metric, figPath=figPath(3))

        dataList.append({'y':y, 'figpath1': figPath(1), 'figpath2': figPath(2), 'figpath3': figPath(3)})


    template = env.get_template("initTemplate.html")

    with open("holyCrap.html", 'w') as f:

        f.write( template.render(itemList = dataList) )

