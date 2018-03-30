from __future__ import division
from sklearn.mixture import GaussianMixture
from utils import read_csv, normalize_data, plot_data
from optimization import convergence_test
from matplotlib import pyplot as plt
from mathematics import *
from scipy.special import logsumexp
import numpy as np
import copy

__author__ = "Ewerton Oliveira"
__copyright__ = "Copyright 2018"
__credits__ = ["Ewerton Lopes"]
__license__ = "MIT"
__version__ = "0.01"
__maintainer__ = "Ewerton Oliveira"
__email__ = "ewerton.lopes@polimi.it"
__status__ = "Production"


class MixGaussBayesStructure:
    """
    Create a variational Bayes mixture of Gaussians model
    """
    def __init__(self, alpha, beta, m, v, W, invW):
        if not invW.size:
            D, D2, K = W.shape
        else:
            D, D2, K = invW.shape

        # store the params
        self.alpha = alpha      # Dirichlet vector parameter (scalar in case of symmetric prior)
        self.beta = beta        #
        self.m = m
        self.v = v
        self.W = np.zeros((D, D, K))
        self.invW = np.zeros((D, D, K))

        for k in range(K):
            if not invW.size:
                self.W[:, :, k] = copy.deepcopy(W[:, :, k])
                self.invW[:, :, k] = np.linalg.inv(W[:, :, k])

            if not W.size:
                self.invW[:, :, k] = copy.deepcopy(invW[:, :, k])
                self.W[:, :, k] = np.linalg.inv(invW[:, :, k])

        # precompute various functions of the distribution for speed

        # E[ln(pi(k))] 10.66
        self.logPiTilde = digamma(alpha) - digamma(np.sum(alpha))

        logdetW = np.zeros((1, K))

        # E[ln(Lambda(:,:,k))]
        self.logLambdaTilde = np.zeros((1, K))

        self.logWishartConst = np.zeros((1, K))

        self.entropy = np.zeros((1, K))

        # B.23
        self.logDirConst = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

        for k in range(K):
            logdetW[:, k] = logdet(self.W[:, :, k])

            # Calculates Bishop's B.81 equation: E[ln|Lambda|]
            self.logLambdaTilde[:, k] = np.sum(digamma(1/2*(v[:, [k]] + 1 - np.arange(1, D+1)))) + \
                                        (D * np.log(2)) + logdetW[:, k]

            logB = -(v[:,k] / 2) * logdetW[:, k] - (v[:, [k]] * D/2) * np.log(2)      \
                      - (D*(D-1)/4) * np.log(np.pi) - np.sum(gammaln(0.5*(v[:, [k]] + 1 - np.arange(1, D+1))))

            # Calculates Bishop's B.79 equation: B(W, v)
            self.logWishartConst[:, k] = -(v[:, [k]]/2) * logdetW[:, k] - (v[:, [k]]*D/2)*np.log(2) - mvtGammaln(D, v[:, [k]]/2)

            assert(np.isclose(logB[0], self.logWishartConst[0, k], rtol=0, atol=1e-2))

            # Calculates Bishop's B.82 equation: H[Lambda]
            self.entropy[:, k] = - self.logWishartConst[:, k] - ((v[:, [k]]-D-1)/2)*self.logLambdaTilde[:, k] + v[:, [k]]*D/2

    def print_params(self):
        # copying parameters
        print 'alpha=\n{}'.format(self.alpha)
        print 'beta=\n{}'.format(self.beta)
        print 'entropy=\n{}'.format(self.entropy)
        print 'invW=\n{}'.format(self.invW[:, :, 0])
        print 'logDirConst=\n{}'.format(self.logDirConst)
        print 'logLambdaTilde=\n{}'.format(self.logLambdaTilde)
        print 'logPiTilde=\n{}'.format(self.logPiTilde)
        print 'logWishartConst=\n{}'.format(self.logWishartConst)
        print 'm=\n{}'.format(self.m)
        print 'v={}'.format(self.v)


def mix_gauss_bayes_fit(X, K, max_iter=200, thresh=1e-5, verbose=True, alpha0=0.001, plotFn=False):

    N, D = X.shape

    model = {}      # TODO: change this container to a better variable other than dict

    # define a vague prior
    alpha = alpha0 * np.ones((1, K))
    m = np.zeros((K, D))
    beta = 1 * np.ones((1, K))       # low precision for mean

    # Sigma = diag(var(X))
    # W = (K^(1/D))*repmat(inv(Sigma), [1,1,K]) # Fraley and Raftery heuristic

    # diag is necessary to make numpy's np.tile replicate Matlab's repmat
    # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
    diag = np.eye(D)
    diag = diag[:, :, np.newaxis]
    W = 200 * np.tile(diag, (1, 1, K))   # W is a D-by-D-by-K matrix

    # v = 5*(D+2)*ones(1,K) # smallest valid dof
    v = 20 * np.ones((1, K))

    model['priorParams'] = MixGaussBayesStructure(alpha, beta, m, v, W, np.array([]))
    model['K'] = K

    # Initialization -- In practice, a better initialization procedure would be to
    # choose the cluster centres mu_k to be equal to a random subset of K data points.
    # It is also worth noting that the K-means algorithm itself is often used to
    # initialize the parameters in a Gaussian mixture model before applying the
    # EM algorithm. Bishop's book pag. 427.

    # fit a Gaussian Mixture Model with two components -- Initialization
    # clf = GaussianMixture(n_components=K, max_iter=30, covariance_type='full', init_params='kmeans',
    #                       tol=0.1, random_state=1)
    # clf.fit(X)
    # xbar = clf.means_          # centres
    # Nk = N*clf.weights_        # priors
    S = np.zeros((D, D, K))    # covariances
    # for k in range(clf.covariances_.shape[0]):    # reshape covariances
    #     S[:, :, k] = clf.covariances_[k]

    Nk = np.array([42.6121, 33.1998, 16.3821, 47.2440, 49.9446, 82.6175])

    xbar = np.array([[0.3665, 0.3326],
                     [-1.1101, - 0.8419],
                     [-1.1632, - 1.5650],
                     [-1.4221, - 1.3396],
                     [0.6977, 1.0827],
                     [0.8791, 0.5886]])

    S[:,:,0] = np.array([[0.0996, 0.0742],
                         [0.0742, 0.2066]])

    S[:,:,1] = np.array([[0.0699, 0.0065],
                         [0.0065, 0.1224]])

    S[:,:,2] = np.array([[0.0251, 0.0104],
                         [0.0104, 0.0514]])

    S[:,:,3] = np.array([[0.0066, 0.0025],
                         [0.0025, 0.1152]])

    S[:,:,4] = np.array([[0.1389, 0.0600],
                         [0.0600, 0.1067]])

    S[:,:,5] = np.array([[0.0517, 0.0113],
                         [0.0113, 0.0749]])

    model['postParams'] = m_step(Nk, xbar, S, model['priorParams'])

    # -- Main loop -- #
    iter_num = 0
    done = False
    log_lik_hist = []
    converged = False

    while not done:
        # E-step
        z, rnk, ll, log_rnk, _ = e_step(model, X)

        # compute responsibilities variables
        Nk, xbar, S = compute_nk_xbar_sk(X, rnk)

        # get lower bound for interation
        log_lik_hist.append(lower_bound(model,  Nk, xbar, S, rnk, log_rnk, iter_num))

        # M-step
        model['postParams'] = m_step(Nk, xbar, S, model['priorParams'])

        p = model['postParams']

        if plotFn:
            plotFn(X, p.alpha, p.m, p.W, p.v, log_lik_hist[iter_num], iter_num)

        # converged?
        if iter_num == 0:
            converged = False
        else:
            converged = convergence_test(log_lik_hist[iter_num], log_lik_hist[iter_num-1], thresh)

        done = converged or (iter_num > max_iter)

        if verbose:
            print 'Iteration #{}, loglik = {}'.format(iter_num, log_lik_hist[iter_num])

        iter_num += 1

    return model, log_lik_hist


def m_step(Nk, xbar, S, priorParams):

    # copying parameters
    alpha0 = copy.deepcopy(priorParams.alpha)
    beta0 = copy.deepcopy(priorParams.beta)
    entropy0 = copy.deepcopy(priorParams.entropy)
    invW0 = copy.deepcopy(priorParams.invW)
    logDirConst0 = copy.deepcopy(priorParams.logDirConst)
    logLambdaTilde0 = copy.deepcopy(priorParams.logLambdaTilde)
    logPiTilde0 = copy.deepcopy(priorParams.logPiTilde)
    logWishartConst0 = copy.deepcopy(priorParams.logWishartConst)
    m0 = copy.deepcopy(priorParams.m)
    v0 = copy.deepcopy(priorParams.v)

    K = alpha0.size
    d = xbar.shape[1]

    alpha = alpha0 + Nk     # Bishop's 10.58
    beta  = beta0 + Nk      # Bishop's 10.60
    m = np.zeros((K, d))
    v = np.zeros((1, K))
    invW = np.zeros((d, d, K))

    for k in range(K):
        if Nk[k] < 0.001:    # extinguished
            m[k, :] = m0[k, :]
            invW[:, :, k] = invW0[:, :, k]
            v[:, [k]] = v0[:, k]
        else:
            m[k, :] = (beta0[:, k] * m0[k, :] + Nk[k] * xbar[k]) / beta[:, [k]]    # Bishop's 10.61

            invW[:, :, k] = invW0[:, :, k] + (Nk[k] * S[:, :, k]) + ((beta0[:, k] * Nk[k]) / (beta0[:, k] + Nk[k])) * \
                            (xbar[[k], :] - m0[[k], :]).transpose().dot(xbar[[k], :] - m0[[k], :])   # Bishop's 10.62

            #W[:,:,k] = np.linalg.inv(invW[:,:,k])
            if np.isnan(np.sum(invW[:, :, k])):
                print 'inverse W has NaN'

            v[:, [k]] = v0[:, k] + Nk[k]    # Bishop's 10.63

    return MixGaussBayesStructure(alpha, beta, m, v, np.array([]), invW)


def compute_nk_xbar_sk(X, resp):
    """
    This computes analogous quantities to the maximum likelihood EM for the Gaussian mixture model.
    In particular, it re-estimates the parameters using the current responsibilities using Bishop's
    equation 10.51; 10.52 and 10.53.
    :param X: the input data.
    :param resp: the responsibilities for each gaussian k on the data X.
    :return:
    """

    K = resp.shape[1]
    d = X.shape[1]

    # Bishop's equation 10.51 (Sum of responsibilities for each k)
    Nk = np.sum(resp, axis=0)

    # NUMERICAL TRICK: Adding a very small number to avoid getting 0 for a responsibility.
    Nk = Nk + 1e-10

    # responsibility weighted mean of x
    Xbar_k = np.zeros((K, d))

    # responsibility weighted covariance of x
    Sk = np.zeros((d, d, K))

    for k in range(K):
        # Bishop's equation 10.52
        Xbar_k[k, :] = np.sum(X * resp[:, [k]], axis=0) / Nk[k]
        XC = X - Xbar_k[k, :]
        # Bishop's equation 10.53
        Sk[:, :, k] = (XC * resp[:, [k]]).conj().transpose().dot(XC) / Nk[k]

    return Nk, Xbar_k, Sk


def e_step(model, X):
    """
    z(i) = argmax_k p(z=k|X(i,:), model) hard clustering
    pz(i,k) = p(z=k|X(i,:), model) soft responsibility
    ll(i) = log p(X(i,:) | model)  log prob of observed data

    Calculate responsibilities using Bishop's equation 10.67

    This is the variational equivalent to the E-step in the EM algorithm.
    Here, we use the current distributions over the model parameters to evaluate
    the moments in Bishop's equations (10.64), (10.65), and (10.66), i.e., the
    expectations need to compute the responsabilities and hence evaluate E[znk] = rnk.

    """

    # copying parameters
    beta = model['postParams'].beta
    log_lambda_tilde = model['postParams'].logLambdaTilde
    log_pi_tilde = model['postParams'].logPiTilde
    m = model['postParams'].m
    v = model['postParams'].v
    W = model['postParams'].W

    K = model['K']
    N, D = X.shape

    E = np.zeros((N, K))

    for k in range(K):
        x_c = X - m[[k], :]     # subtract mean

        # Bishop's 10.64. The sum is a row-wise sum.
        E[:, [k]] = (D / beta[:, [k]]) + v[:, [k]] * np.sum(x_c.dot(W[:, :, k]) * x_c, axis=1, keepdims=True)

    log_rho = np.tile(log_pi_tilde + 0.5*log_lambda_tilde, (N, 1)) - 0.5 * E

    log_sum_rho = logsumexp(log_rho, axis=1).reshape((-1, 1))   # the reshape make the result a col-vector
    log_r = log_rho - np.tile(log_sum_rho, (1, K))
    r = np.exp(log_r)
    nk = np.exp(logsumexp(log_r, axis=0))
    z = log_r.max(1)    # max element of each row

    return z, r, log_sum_rho, log_r, nk


def lower_bound(model,  Nk, xbar, S, rnk, logrnk, iter_num):
    """
    Evaluates the lower bound.
    From Bishop's book section 10.2.2: We can also straightforwardly evaluate
    the lower bound in equation (10.3) for this model. In practice, it is
    useful to be able to monitor the bound during the re-estimation in order
    to test for convergence. It can also provide a valuable check on both the
    mathematical expressions for the solutions and their software implementation,
    because at each step of the iterative re-estimation procedure the value of
    this bound should not decrease. We can take this a stage further to provide
    a deeper test of the correctness of both the mathematical derivation of the
    update equations and of their software implementation by using finite
    differences to check that each update does indeed give a (constrained) maximum
    of the bound (Svensen and Bishop, 2004).
    :param model:
    :param Nk:
    :param xbar:
    :param S:
    :param rnk:
    :param logrnk:
    :param iter_num:
    :return:
    """

    # copying postParams
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

    # copying priorParams
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

    # Bishop's equation 10.71
    ElogpX = np.zeros((1, K))
    for k in range(K):
        xbarc = xbar[[k], :] - m[[k], :]

        # the reshape maintains the 1x2 for after the multiplications
        ElogpX[:, k] = 0.5 * Nk[k]                                     \
                      * (logLambdaTilde[:, k] - (D/beta[:, [k]])
                         - np.trace(v[:, [k]] * S[:, :, k].dot(W[:, :, k]))
                         - v[:, [k]] * np.sum(xbarc.dot(W[:, :, k]) * xbarc, keepdims=True)
                         - (D * np.log(2*np.pi)))    # Bishop's equation 10.71
    ElogpX = np.sum(ElogpX)

    # Bishop's equation 10.72
    ElogpZ = np.sum(Nk * logPiTilde)

    # Bishop's equation 10.73
    Elogppi = logDirConst0 + np.sum((alpha0-1)*logPiTilde)

    # Bishop's equation 10.74
    ElogpmuSigma = np.zeros((1, K))
    for k in range(K):
        mc = m[[k], :] - m0[[k], :]

        ElogpmuSigma[:, k] = 0.5 * (D * np.log(beta0[:, k]/(2*np.pi))
                               + logLambdaTilde[:, k] - ((D*beta0[:, k])/beta[:, [k]])
                               - beta0[:, k]*v[:, [k]]*np.sum(mc.dot(W[:, :, k]) * mc, keepdims=True)) \
                               + logWishartConst0[:, k]                                \
                               + 0.5 * (v0[:, k] - D - 1) * logLambdaTilde[:, k]                 \
                               - 0.5 * v[:, [k]] * np.trace(invW0[:, :, k].dot(W[:, :, k]))

    ElogpmuSigma = np.sum(ElogpmuSigma)

    # Entropy terms
    # Bishop's equation 10.75
    ElogqZ = np.sum(np.sum(rnk*logrnk, axis=0, keepdims=True))

    # Bishop's equation 10.76
    Elogqpi = np.sum((alpha - 1) * logPiTilde) + logDirConst

    # Bishop's equation 10.77
    ElogqmuSigma = np.sum((1/2)*logLambdaTilde + (D/2)*np.log(beta / (2*np.pi)) - (D/2) - entropy)

    # Overall sum
    # Bishop's equation 10.70
    L = ElogpX + ElogpZ + Elogppi + ElogpmuSigma - ElogqZ - Elogqpi - ElogqmuSigma

    if np.isnan(L):
      print "ElogpX: {}\nElogpZ: {}\nElogppi: {}\n\
             ElogpmuSigma: {}\nElogqZ: {}\n\
             Elogqpi: {}\nElogqmuSigma: {}\n".format(ElogpX, ElogpZ,
             Elogppi, ElogpmuSigma, ElogqZ, Elogqpi, ElogqmuSigma)
    return L


def main(K=6):

    # set seed
    np.random.seed(1)

    # set numpy print options (pretty print matrix)
    np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

    # load Data
    data = read_csv()

    # preprocess
    X = normalize_data([d[0] for d in data])
    Y = normalize_data([d[1] for d in data])
    data = np.array([X, Y]).transpose()

    # run mixGaussBayesFit
    model, loglikHist = mix_gauss_bayes_fit(data, K)

    # plot likelihood
    plt.plot(loglikHist, '-', marker='*', lw=3)
    plt.yticks(np.arange(-1100, -601,50))
    plt.xlim([0,100])
    plt.xlabel('iterations')
    plt.ylabel('lower bound on log marginal likelihood')
    plt.title('variational Bayes objective for GMM on old faithful data')
    plt.show()

if __name__ == '__main__':
    main()
