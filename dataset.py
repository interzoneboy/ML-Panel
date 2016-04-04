"""

Functions to handle all the data fetching/cleaning/normalizing/splitting/etc.
These will produce a uniform tuple output for the algorithms to make use of.
All messy data details are sealed off from the algorithms, pipelining, and 
visualization code, in this module.

"""
from scipy import linalg
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
from sklearn import preprocessing

# These lines break Ipython autocomplete for some reason.
#
#from IPython.terminal.embed import InteractiveShellEmbed
#ipshell = InteractiveShellEmbed(banner1="dataset breakpoint")



def read_data_csv(filename, *args, **kwargs):
    """ Read <filename> and return a pandas data frame.

    :param filename: Filename to read.
    :type filename: str
    """
    return pd.read_csv(filename)


def randomSplit(df, y_var_name, x_var_names, testSize=0.35, seedIn=None):
    """ Scale, impute missing, then split dataset *df* randomly into tuple (test_x, test_y, train_x, train_y)

    :param df: the pandas data frame to split
    :param y_var_name: the name of the variable (column of df) we try to predict.
    :type y_var_name: str
    :param x_var_names: list of the predictor variables.
    :type x_var_names: list(str)
    :param testSize: the fraction (0.0 to 1.0) of the dataset to put in TEST partition.
    :param seedIn: seed to the random number generator for reproducible results (what ARE those even).
    :returns: dict -- {'train_x','train_y','test_x','test_y'}
    """
    scaler = preprocessing.StandardScaler()
    imputer = preprocessing.Imputer(missing_values="NaN", strategy="mean", axis=0)

    # remove rows where y_variable is missing.
    good_inds = (np.isnan(df[y_var_name])==False).nonzero()
    d = df.iloc[good_inds]

    if seedIn!=None:
        ss = ShuffleSplit(d.shape[0], n_iter=1, test_size=testSize, random_state=seedIn)
    else:
        ss = ShuffleSplit(d.shape[0], n_iter=1, test_size=testSize)

    training_inds, test_inds = ss.__iter__().next()
    training_rows = d.iloc[training_inds]
    test_rows = d.iloc[test_inds]

    data_tr = training_rows[x_var_names]
    imputer.fit(data_tr)
    scaler.fit(imputer.transform(data_tr))
    data_tr_scaled = scaler.transform(imputer.transform(data_tr))

    data_test = test_rows[x_var_names]
    data_test_scaled = scaler.transform(imputer.transform(data_test))

    return {'test_x': data_test_scaled,
            'test_y': test_rows[y_var_name],
            'train_x': data_tr_scaled,
            'train_y': training_rows[y_var_name],
            }



def compute_quantiles(df, y_var_name, quantile, y_var_pref="qtile_"):
    """ Return a copy of df with a new column (y_var_pref + quantile + y_var_name) that
        reflects y_var_name broken up into the appropriate quantiles.

    :param df: pandas data frame to add quantile column.
    :param y_var_name: the column name of *df* for which we want the quantiles.
    :param quantile: int -- the number of quantiles bins needed.
    :param y_var_pref: prefix to append to the new column name, which is otherwise formed from the quantile
        number and the y_var_name.
    :returns: *copy* of df, with the new column added.
    """
    pass


def top_or_bottom(df, y_var_quantile_name, y_var_pref="TB"):
    """ Return a copy of df, with a new binary valued column, indicating whether the corresponding quantile
        is in the top or the bottom quantile. Non-top/bottom entries are left as NA/NaN. This is a useful function
        combined with compute_quantiles for generating quantile classification target variables.

    :param df: pandas data frame to add Top/Bottom of quantile column
    :param y_var_quantile_name: The name of the quantile column that already exists
    :param y_var_pref: The prefix to append to the new column name (taken from y_var_quantile_name otherwise)
    :returns: *copy* of the data frame df, with a binary (+ NaN) valued top/bottom quantile column.
    """
    pass


def group_dataset(df, selection_function):
    """ Filter *df* by only including rows where grp_function returns True.

    :param df: pandas data frame to be filtered.
    :param selection_function: a function that, when called like select_func(df) will return row indices to keep, and 
        when called like select_func() will return a string identifying the filtering operation (eg) "Female, eth=J")
    :returns: the filtered dataset.
    """
    pass
