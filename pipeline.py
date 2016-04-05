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



