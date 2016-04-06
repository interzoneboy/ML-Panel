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


if __name__ == "__main__":

    d = DS.read_data_csv("SampleData/example_data.csv")
    d2 = DS.compute_quantiles(d, "y_var_3", 4)
    d3 = DS.top_or_bottom(d2, "qtile_4_y_var_3", collapse=True)

    x_names = [a for a in d3.columns if ('x_var' in a.lower() and ('agg' not in a.lower()))]
    y_name = "TBqtile_4_y_var_3"

    dReady = DS.randomSplit(d3, y_name, x_names)

