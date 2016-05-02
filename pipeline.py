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
from contextlib import contextmanager

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


def run_test():
    d = DS.read_data_csv("SampleData/example_data.csv")
    d2 = DS.compute_quantiles(d, "y_var_3", 4)
    d3 = DS.top_or_bottom(d2, "qtile_4_y_var_3", collapse=True)
    
    x_names = [a for a in d3.columns if ('x_var' in a.lower() and ('agg' not in a.lower()))]
    y_name = "TBqtile_4_y_var_3"
    
    
    alg_results = {}
    for kk in ['tree']:
        print "Working on " + str(kk)
        alg_results[kk] = []
        for numIter in range(0, 500):
            dReady = DS.randomSplit(d3, y_name, x_names)
            alg_results[kk].append(ALG.algorithms[kk](dReady))
            if numIter%20==0:
                print "Finished %s iterations." % (str(numIter),)

    return alg_results


def getPickle():

    with open("alg_results.saved", 'r') as f:
        output = pickle.load(f)
    return output

def savePickle(p):

    with open("alg_results.saved", 'w') as f:
        pickle.dump(p, f)


def runMain():
#if __name__ == "__main__":

    #res = run_test()
    d = DS.read_data_csv("SampleData/example_data.csv")
    d2 = DS.compute_quantiles(d, "y_var_3", 4)
    d3 = DS.top_or_bottom(d2, "qtile_4_y_var_3", collapse=True)
    
    x_names = [a for a in d3.columns if ('x_var' in a.lower() and ('agg' not in a.lower()))]
    y_name = "TBqtile_4_y_var_3"
     
    alg_results = {}
    for (kk,rr) in [('svm_linear',0.008), ('svm_rbf',0.5), ('lda',None), ('qda',0.2), ('logistic',0.001), ('SGDlogistic',0.0000001)]:
        print "Working on " + str(kk)
        alg_results[kk] = []
        for numIter in range(0, 100):
            dReady = DS.randomSplit(d3, y_name, x_names)
            alg_results[kk].append(ALG.algorithms[kk](dReady, regParam=rr))
            #alg_results[kk].append(ALG.algorithms[kk](dReady))
            if numIter%20==0:
                print "Finished %s iterations." % (str(numIter),)

    return (alg_results, x_names, y_name)
