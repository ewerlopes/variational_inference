from __future__ import division
from scipy.special import psi, gammaln
import numpy as np

def digamma(X):
    '''
    Digamma function: d/dx log gamma(X)
    It returns digamma(x) = d log(gamma(X)) / dx
    If X is a matrix, returns the digamma function evaluated at each element.
    Parameters:
        x : a matrix, vector or scalar
    Returns:
        digamma(x) = d log(gamma(X)) / dx
    '''
    return psi(X)


def logdet(A):
    '''
    Computes log(det(A)) where A is positive-definite
    This is faster and more stable than using log(det(A)).
    PMTKauthor Tom Minka(c) Microsoft Corporation. All rights reserved.

    Parameters:
        A: positive-definite matrix.
    Return:
        log(det(A)) or None in case matrix is note positive-definite
    '''

    #'LogDet: shape of A is\n{}'.format(A)

    try:
        U = np.linalg.cholesky(A).T
        y = 2 * np.sum(np.log(np.diag(U)))
        return y
    except Exception as e:
        print 'Matrix is not positive definite'
        return 0

def mvtGammaln(n, alpha):
    '''
    Returns the log of multivariate gamma(n, alpha) value.
    necessary for avoiding underflow/overflow problems that alpha > (n-1)/2

    See Muirhead pp 61-62.
    '''
    return ((n*(n-1))/4)*np.log(np.pi)+np.sum(gammaln(alpha+0.5*(1-np.arange(1,n+1))))
