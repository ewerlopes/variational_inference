from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from mathematics import *
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from optimization import convergenceTest

def read_csv(filename='oldFaith.txt'):
    f = open(filename, 'r')
    data = []
    for line in f.readlines():
        line = line.strip().split(' ')
        data.append(tuple([float(line[0]),int(line[1])]))
    return data

def normalize_data(X):
    return (np.array(X) - np.mean(X)) / np.var(X)

def plotData(X,Y,normalize=False):
    fig = plt.figure(figsize=(4,2))

    if normalize:
        X = normalize_data(X)
        Y = normalize_data(Y)

    plt.scatter(X,Y)
    plt.show()


class Model:
    '''Create a variational Bayes mixture of Gaussians model'''

    def __init__(alpha, beta, entropy, invW, logDirConst,
                logLambdaTilde, logPiTilde, logWishartConst, m, v, W):
        self.alpha = alpha
        self.beta  = beta
        self.entropy = entropy
        self.invW = invW
        self.logDirConst = logDirConst
        self.logLambdaTilde = logLambdaTilde
        self.logPiTilde = logPiTilde
        self.logWishartConst = logWishartConst
        self.m = m
        self.v = v
        self.W = W
        self.modelType = 'mixGaussBayes'


class mixGaussBayesStructure:
    def __init__(self, alpha, beta, m, v, W, invW):
        if not invW.size:
            D, D2, K = W.shape
        else:
            D, D2, K = invW.shape

        # store the params
        self.alpha = alpha
        self.beta = beta
        self.m = m
        self.v = v
        self.W = np.zeros((D,D,K))
        self.invW = np.zeros((D,D,K))

        for k in range(K):
            if not invW.size:
                self.W[:,:,k] = W[:,:,k]
                self.invW[:,:,k] = np.linalg.inv(W[:,:,k])

            if not W.size:
                self.invW[:,:,k] = invW[:,:,k]
                self.W[:,:,k] = np.linalg.inv(invW[:,:,k])

        #precompute various functions of the distribution for speed'''

        # E[ln(pi(k))] 10.66
        self.logPiTilde = digamma(alpha) - digamma(np.sum(alpha))

        logdetW = np.zeros((1,K))

        # E[ln(Lambda(:,:,k))]
        self.logLambdaTilde = np.zeros((1,K))

        self.logWishartConst = np.zeros((1,K))

        self.entropy = np.zeros((1,K))

        # B.23
        self.logDirConst = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

        for k in range(K):
            logdetW[:,k] = logdet(self.W[:,:,k])

            # Calculates Bishop's B.81 equation: E[ln|Lambda|]
            self.logLambdaTilde[:,k] = np.sum(digamma(1/2*(v[:,k] + 1 - np.arange(1,D+1)))) + (D * np.log(2))  + logdetW[:,k]

            logB = -(v[:,k] / 2) * logdetW[:,k] - (v[:,k] * D/2) * np.log(2)      \
                      -(D*(D-1)/4) * np.log(np.pi) - np.sum(gammaln(0.5*(v[:,k] + 1 - np.arange(1,D+1))))

            # Calculates Bishop's B.79 equation: B(W, v)
            self.logWishartConst[:,k] = -(v[:,k]/2) * logdetW[:,k] -(v[:,k]*D/2)*np.log(2) - mvtGammaln(D,v[:,k]/2)

            assert(np.isclose(logB[0], self.logWishartConst[0,k], rtol=0, atol=1e-2))

            # Calculates Bishop's B.82 equation: H[Lambda]
            self.entropy[:,k] = -self.logWishartConst[:,k] - (v[:,k]-D-1)/2*self.logLambdaTilde[:,k] + v[:,k]*D/2

            #params.logLambdaTilde(k) = wishartExpectedLogDet(params.W(:,:,k), v(k), logdetW(k))
            #params.entropy(k) = wishartEntropy(params.W(:,:,k), v(k), logdetW(k))
            #params.logWishartConst(k) = wishartLogConst(params.W(:,:,k), v(k), logdetW(k))


def mixGaussBayesFit(X, K, maxIter=200, thresh=1e-5, verbose=False,
                    alpha0= 0.001, plotFn=False):

    N, D = X.shape

    model = {}      #TODO change this container to a better variable other than dict

    ## define a vague prior
    alpha = alpha0 * np.ones((1,K))
    m = np.zeros((K,D))
    beta = 1 * np.ones((1,K))       # low precision for mean

    #Sigma = diag(var(X))
    #W = (K^(1/D))*repmat(inv(Sigma), [1,1,K]) # Fraley and Raftery heuristic

    # diag is necessary to make numpy's np.tile replicate Matlab's repmat
    # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
    diag = np.eye(D)
    diag = diag[:,:,np.newaxis]
    W = 200 * np.tile(diag,(1, 1, K))   # W is a D-by-D-by-K matrix

    #v = 5*(D+2)*ones(1,K) # smallest valid dof
    v = 20 * np.ones((1,K))

    model['priorParams'] = mixGaussBayesStructure(alpha, beta, m, v, W, np.array([]))
    model['K'] = K

    ## Initialization -- In practice, a better initialization procedure would be to
    # choose the cluster centres mu_k to be equal to a random subset of K data points.
    # It is also worth noting that the K-means algorithm itself is often used to
    # initialize the parameters in a Gaussian mixture model before applying the
    # EM algorithm. Bishop's book pag. 427.

    # fit a Gaussian Mixture Model with two components -- Initialization
    clf = GaussianMixture(n_components=K, max_iter=30, covariance_type='full',
                        init_params='kmeans', tol=0.1, random_state=1)
    clf.fit(X)
    xbar = clf.means_       # centres
    Nk = N*clf.weights_     # priors

    S = np.zeros((D,D,K))   # covariances
    for k in range(clf.covariances_.shape[0]): # reshape it
        S[:,:,k] = clf.covariances_[k]


    np.set_printoptions(precision=3, suppress=True)

    model['postParams'] = Mstep(Nk, xbar, S, model['priorParams'])

    ### -- Main loop -- ###
    iter_num = 0
    done = False
    loglikHist = []
    converged = False
    while not done:
        # E step
        z, rnk, ll, logrnk, _ = mixGaussBayesInfer(model, X)
        Nk, xbar, S = computeEss(X, rnk)
        loglikHist.append(lowerBound(model,  Nk, xbar, S, rnk, logrnk, iter_num))

        # M step
        model['postParams'] = Mstep(Nk, xbar, S, model['priorParams'])

        p = model['postParams']
        if plotFn:
            plotFn(X, p.alpha, p.m, p.W, p.v, loglikHist[iter_num], iter_num) # TODO

        # Converged?
        if iter_num == 0:
            converged = False
        else:
            converged = convergenceTest(loglikHist[iter_num], loglikHist[iter_num-1], thresh) # TODO

        done = converged or (iter_num > maxIter)

        if verbose:
            print 'Iteration #{}, loglik = {:.2}}'.format(iter_num, loglikHist[iter_num])

        iter_num +=1

    return model, loglikHist


def Mstep(Nk, xbar, S, priorParams):

    # copying parameters
    alpha0 = priorParams.alpha
    beta0 = priorParams.beta
    entropy0 = priorParams.entropy
    invW0 = priorParams.invW
    logDirConst0 = priorParams.logDirConst
    logLambdaTilde0 = priorParams.logLambdaTilde
    logPiTilde0 = priorParams.logPiTilde
    logWishartConst0 = priorParams.logWishartConst
    m0 = priorParams.m
    v0 = priorParams.v

    K = alpha0.size
    d = xbar.shape[1]

    alpha = alpha0 + Nk     # Bishop's 10.58
    beta  = beta0 + Nk      # Bishop's 10.60
    m = np.zeros((K,d))
    v = np.zeros((1,K))
    invW = np.zeros((d,d,K))

    for k in range(K):
        if Nk[k] < 0.001: # extinguished
            m[k,:] = m0[k,:]
            invW[:,:,k] = invW0[:,:,k]
            v[:,k] = v0[:,k]
        else:
            m[k,:] = (beta0[:,k] * m0[k,:] + Nk[k] * xbar[k]) / beta[:,k] # 10.61

            invW[:,:,k] = invW0[:,:,k] + Nk[k] * S[:,:,k] + \
                          (beta0[:,k]* Nk[k] / (beta0[:,k] + Nk[k])) * \
                          (xbar[k] - m0[k,:]).transpose().dot(xbar[k] - m0[k,:]) # 10.62

            #W[:,:,k] = np.linalg.inv(invW[:,:,k])
            if np.isnan(np.sum(invW[:,:,k])):
                print 'inverse W has NaN'
            v[:,k] = v0[:,k] + Nk[k] # 10.63

    return  mixGaussBayesStructure(alpha, beta, m, v, np.array([]), invW)


def wishartLogConst(W, v, logdetW=None): # Bishop's B.79
    d = W.shape[0]
    if logdetW is None:
        logdetW = logdet(W)
    return -(v/2)*logdetW -(v*d/2)*log(2) - mvtGammaln(d,v/2)


def wishartEntropy(W, v, logdetW=None): # Bishop's  B.82
    d = W.shape[0]
    if logdetW is None:
      logdetW = logdet(W)
    return - wishartLogConst(W, v, logdetW)       \
           - (v-d-1)/2*wishartExpectedLogDet(W, v, logdetW) + v*d/2


def wishartExpectedLogDet(W, v, logdetW): # Bishop's B.81
    d = W.shape[0]
    if logdetW is None:
        logdetW = logdet(W)
    return sum(digamma(1/2*(v + 1 - np.arange(1,d+1)))) + d*log(2)  + logdetW


def computeEss(X, weights):
    # weights(n,k)
    K = weights.shape[1]
    d = X.shape[1]
    Nk = np.sum(weights, axis=0) # 10.51
    Nk = Nk + 1e-10
    SSxbar = np.zeros((K,d))
    SSXX = np.zeros((d,d,K))

    for k in range(K):    #the .reshape(-1,1) makes the vector a col-vector.
        SSxbar[k,:] = np.sum(X * weights[:,k].reshape(-1,1), axis=0) / Nk[k] # 10.52
        XC = X - SSxbar[k,:]
        SSXX[:,:,k] = (XC * weights[:,k].reshape(-1,1)).transpose().dot(XC) / Nk[k] # 10.53
    return (Nk, SSxbar, SSXX)


def mixGaussBayesInfer(model, X):
    '''
    z(i) = argmax_k p(z=k|X(i,:), model) hard clustering
    pz(i,k) = p(z=k|X(i,:), model) soft responsibility
    ll(i) = log p(X(i,:) | model)  logprob of observed data

    Calculate responsibilities using Bishop's equation 10.67
    '''

    # copying parameters
    alpha = model['postParams'].alpha
    beta = model['postParams'].beta
    entropy = model['postParams'].entropy
    invW = model['postParams'].invW
    logDirConst = model['postParams'].logDirConst
    logLambdaTilde = model['postParams'].logLambdaTilde
    logPiTilde = model['postParams'].logPiTilde
    logWishartConst = model['postParams'].logWishartConst
    m = model['postParams'].m
    v = model['postParams'].v
    W = model['postParams'].W

    K = model['K']
    N, D = X.shape

    E = np.zeros((N,K))

    for k in range(K):
        XC = X - m[k,:]     # subtract mean
        E[:,k]  = D / (beta[:,k]) + v[:,k] * np.sum(XC.dot(W[:,:,k]) * XC, axis=1) # Bishop's 10.64. The sum is a row-wise sum.

    logRho = np.tile(logPiTilde + 0.5*logLambdaTilde,(N, 1)) - 0.5 * E
    logSumRho = logsumexp(logRho,axis=1).reshape((-1, 1))   # the reshape make the result a col-vector
    logr = logRho - np.tile(logSumRho,(1,K))
    r = np.exp(logr)
    Nk = np.exp(logsumexp(logr, axis=0))
    z = logr.max(1) # max element of each row

    return (z, r, logSumRho, logr, Nk)


def lowerBound(model,  Nk, xbar, S, rnk, logrnk, iter):
    # Bishop's book section 10.2.2

    ## copying postParams
    alpha = model['postParams'].alpha
    beta = model['postParams'].beta
    entropy = model['postParams'].entropy
    invW = model['postParams'].invW
    logDirConst = model['postParams'].logDirConst
    logLambdaTilde = model['postParams'].logLambdaTilde
    logPiTilde = model['postParams'].logPiTilde
    logWishartConst = model['postParams'].logWishartConst
    m = model['postParams'].m
    v = model['postParams'].v
    W = model['postParams'].W

    ## copying priorParams
    alpha0 = model['priorParams'].alpha
    beta0 = model['priorParams'].beta
    entropy0 = model['priorParams'].entropy
    invW0 = model['priorParams'].invW
    logDirConst0 = model['priorParams'].logDirConst
    logLambdaTilde0 = model['priorParams'].logLambdaTilde
    logPiTilde0 = model['priorParams'].logPiTilde
    logWishartConst0 = model['priorParams'].logWishartConst
    m0 = model['priorParams'].m
    v0 = model['priorParams'].v
    W0 = model['priorParams'].W

    D, D2, K = W.shape

    # 10.71
    ElogpX = np.zeros((1,K))

    for k in range(K):
        xbarc = xbar[k,:] - m[k,:]

        # the reshape maintains the 1x2 for after the multiplications
        ElogpX[:,k] = 0.5 * Nk[k]                                     \
                      * (logLambdaTilde[:,k] - D/beta[:,k]            \
                      - np.trace((v[:,k]*S[:,:,k]).dot(W[:,:,k]))     \
                      - v[:,k] * np.sum((xbarc.dot(W[:,:,k]) * xbarc)
                                         .reshape((1,-1)), axis=1)    \
                      - D * np.log(2*np.pi)) # 10.71

    ElogpX = np.sum(ElogpX)

    #10.72
    ElogpZ = np.sum(Nk * logPiTilde)

    # 10.73
    Elogppi = logDirConst0 + np.sum((alpha0-1)*logPiTilde)

    #10.74
    ElogpmuSigma = np.zeros((1,K))
    for k in range(K):
        mc = m[k,:] - m0[k,:]
        #logB0(k) = (v0(k)/2)*logdet(invW0(:,:,k)) - (v0(k)*D/2)*log(2) ...
        #        - (D*(D-1)/4)*log(pi) - sum(gammaln(0.5*(v0(k)+1 -[1:D])))

        ElogpmuSigma[:,k] = 0.5*(D*np.log(beta0[:,k]/(2*np.pi))                         \
                               + logLambdaTilde[:,k] - D*beta0[:,k]/beta[:,k]         \
                               - beta0[:,k]*v[:,k]*np.sum((mc.dot(W[:,:,k]) * mc)
                                                           .reshape((1,-1)), axis=1))  \
                               + logWishartConst0[:,k]                                \
                          + 0.5*(v0[:,k] - D - 1)*logLambdaTilde[:,k]                 \
                          - 0.5*v[:,k]*np.trace(invW0[:,:,k].dot(W[:,:,k]))

    ElogpmuSigma = np.sum(ElogpmuSigma)

    # Entropy terms
    #10.75
    #ElogqZ = sum(sum(rnk.*log(rnk)))
    ElogqZ = np.sum(np.sum(rnk*logrnk,axis=1))

    #10.76
    Elogqpi = np.sum((alpha - 1) * logPiTilde) + logDirConst

    #10.77
    #TODO verify whether there's a diff between / and ./ in MATLAB
    ElogqmuSigma = np.sum(1/2*logLambdaTilde + D/2*np.log(beta /(2*np.pi)) - D/2 - entropy)

    # Overall sum
    # 10.70
    L = ElogpX + ElogpZ + Elogppi + ElogpmuSigma - ElogqZ - Elogqpi - ElogqmuSigma

    if np.isnan(L):
      print "ElogpX: {}\nElogpZ: {}\nElogppi: {}\n\
             ElogpmuSigma: {}\nElogqZ: {}\n\
             Elogqpi: {}\nElogqmuSigma: {}\n".format(ElogpX, ElogpZ,
             Elogppi, ElogpmuSigma, ElogqZ, Elogqpi, ElogqmuSigma)
    return L

def run_vbem(K=6):
    ## Load Data
    np.random.seed(0)
    data = read_csv()
    X = normalize_data([d[0] for d in data])
    Y = normalize_data([d[1] for d in data])
    data = np.array([X,Y]).transpose()

    ## Run mixGaussBayesFit
    model, loglikHist = mixGaussBayesFit(data,K)

    print loglikHist

    ## Plot
    # figure()
    # plot(loglikHist, 'o-', 'linewidth', 3)
    # xlabel('iter')
    # ylabel('lower bound on log marginal likelihood')
    # title('variational Bayes objective for GMM on old faithful data')



if __name__ == '__main__':
    run_vbem()
