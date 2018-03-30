from __future__ import division
import numpy as np
import sys


def convergence_test(fval, previous_fval, threshold=1e-4, warn=True):
    """
    Check if an objective function has converged

    We have converged if the slope of the function falls below 'threshold',
    i.e., |f(t) - f(t-1)| / avg < threshold, where avg = (|f(t)| + |f(t-1)|)/2
    'threshold' defaults to 1e-4.

    This stopping criterion is from Numerical Recipes in C p423.
    """

    converged = False
    delta_fval = np.abs(fval - previous_fval)
    avg_fval = (np.abs(fval) + np.abs(previous_fval) + np.spacing(1))/2
    if (delta_fval / avg_fval) < threshold:
        converged = True

    if warn and (fval-previous_fval) < -2*np.spacing(1):    # fval < previous_fval
        print >> sys.stderr.write('WARNING: objective decreased when testing convergence!\n')

    return converged
