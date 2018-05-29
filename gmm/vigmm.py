from __future__ import division
from sklearn.mixture import GaussianMixture
from utils import read_csv, normalize_data, plot_data
from vi_math.optimization import convergence_test
from matplotlib import pyplot as plt
from vi_math.mathematics import *
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


class BaseParameters:
    """
    Holds the parameters for the VBGMM.
    """

    def __init__(self, alpha, beta, mean, dof, s_mtx, inv_s_mtx):
        if not inv_s_mtx.size:
            self.D, self.D2, self.K = s_mtx.shape
        else:
            self.D, self.D2, self.K = inv_s_mtx.shape

        self.alpha = alpha  # Dirichlet vector parameter (scalar in case of symmetric prior)
        self.beta = beta  # precision (Bishop's 'beta' parameter)
        self.mean = mean  # mean (Bishop's 'm' parameter)
        self.dof = dof  # Wishart 'degrees of freedom' (Bishop's 'v' parameter)
        self.s_mtx = np.zeros((self.D, self.D, self.K))  # Wishart Scale matrix (Bishop's 'W' parameter)
        self.inv_s_mtx = np.zeros((self.D, self.D, self.K))  # Wishart 'degrees of freedom' (Bishop's 'invW' parameter)

        for k in range(self.K):
            if not inv_s_mtx.size:  # in case inv_s_mtx is empty
                self.s_mtx[:, :, k] = copy.deepcopy(s_mtx[:, :, k])
                self.inv_s_mtx[:, :, k] = np.linalg.inv(s_mtx[:, :, k])

            if not s_mtx.size:  # in case s_mtx is empty
                self.inv_s_mtx[:, :, k] = copy.deepcopy(inv_s_mtx[:, :, k])
                self.s_mtx[:, :, k] = np.linalg.inv(inv_s_mtx[:, :, k])


class Estimate(BaseParameters):

    def __init__(self, alpha, beta, mean, dof, s_mtx, inv_s_mtx):
        BaseParameters.__init__(self,alpha, beta, mean, dof, s_mtx, inv_s_mtx)
        self.log_pi_tilde = None  # Bishop's equation 10.66 : E[ln(pi_k)]
        self.log_lambda_tilde = None  # Bishop's equation 10.65/B.81 : E[ln|Lambda_k|]
        self.exp_mu_lambda = None  # Bishop's equation 10.64
        self.log_wishart_const = None
        self.log_dirichlet_const = None
        self.entropy = None

    def get_log_wishart_const(self):
        # constants
        log_wishart_const = np.zeros((1, self.K))

        for k in range(self.K):
            log_det_s_mtx = logdet(self.s_mtx[:, :, k])

            log_B = -(self.dof[:, k] / 2) * log_det_s_mtx - (self.dof[:, [k]] * self.D / 2) * np.log(2) \
                    - (self.D * (self.D - 1) / 4) * np.log(np.pi) - \
                    np.sum(gammaln(0.5 * (self.dof[:, [k]] + 1 - np.arange(1, self.D + 1))))

            # Calculates Bishop's B.79 equation: B(W, v)
            log_wishart_const[:, k] = -(self.dof[:, [k]] / 2) * log_det_s_mtx - \
                                      (self.dof[:, [k]] * self.D / 2) * np.log(2) - \
                                      mvtGammaln(self.D, self.dof[:, [k]] / 2)

            assert (np.isclose(log_B, log_wishart_const[:, k], rtol=0, atol=1e-2))
        return log_wishart_const

    def get_log_dir_const(self):
        # Bishop's equation B.23
        return gammaln(np.sum(self.alpha)) - np.sum(gammaln(self.alpha))

    def get_expectations(self, data):
        N, D = data.shape

        self.exp_mu_lambda = np.zeros((N, self.K))
        for k in range(self.K):
            x_c = data - self.mean[[k], :]  # subtract mean
            self.exp_mu_lambda[:, [k]] = (D / self.beta[:, [k]]) + self.dof[:, [k]] * \
                                         np.sum(x_c.dot(self.s_mtx[:, :, k]) * x_c, axis=1, keepdims=True)

        self.log_lambda_tilde = np.zeros((1, self.K))
        for k in range(self.K):
            log_det_s_mtx = logdet(self.s_mtx[:, :, k])
            # Calculates Bishop's B.81 equation:
            self.log_lambda_tilde[:, k] = np.sum(digamma(1 / 2 * (self.dof[:, [k]] + 1 - np.arange(1, D + 1)))) \
                                          + (D * np.log(2)) + log_det_s_mtx

        self.log_pi_tilde = digamma(self.alpha) - digamma(np.sum(self.alpha))

        return self.exp_mu_lambda, self.log_lambda_tilde, self.log_pi_tilde

    def __str__(self):
        """Print all parameters"""
        print 'Alpha=\n{}'.format(self.alpha)
        print 'Beta=\n{}'.format(self.beta)
        print 'Entropy=\n{}'.format(self.entropy)
        print 'Wishart Inverse Scale Matrix=\n{}'.format(self.s_mtx[:, :, 0])
        print 'Log of Dirichlet constant=\n{}'.format(self.log_dirichlet_const)
        print 'Log of LambdaTilde=\n{}'.format(self.log_lambda_tilde)
        print 'Log of PiTilde=\n{}'.format(self.log_pi_tilde)
        print 'Log of Wishart constant=\n{}'.format(self.log_wishart_const)
        print 'Mean=\n{}'.format(self.mean)
        print 'Wishart DOF={}'.format(self.dof)


class VBGMM:
    """Gaussian Mixture Model with Variational Bayesian (VB) Learning"""

    def __init__(self, data, K, verbose=True, alpha0=0.001, plotFn=False):
        self.data = data
        self.K = K
        self.verbose = verbose
        self.alpha0 = alpha0
        self.plotFn = plotFn
        self.N, self.data_dim = data.shape
        self.prior_params = None
        self.post_params = None

    def _get_entropy(self):
        pass

    def _init_prior(self):
        """Initialize prior parameters"""
        # define a vague prior
        alpha = self.alpha0 * np.ones((1, self.K))
        m = np.zeros((self.K, self.data_dim))
        beta = 1 * np.ones((1, self.K))  # low precision for mean
        # diag is necessary to make numpy's np.tile replicate Matlab's repmat
        # https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
        diag = np.eye(self.data_dim)
        diag = diag[:, :, np.newaxis]
        W = 200 * np.tile(diag, (1, 1, self.K))  # W is a D-by-D-by-K matrix
        v = 20 * np.ones((1, self.K))
        self.prior_params = Estimate(alpha, beta, m, v, W, np.array([]))
        _, _, _ = self.prior_params.get_expectations(self.data)

    def _init_posterior(self, Nk, xbar, S):
        self.post_params = self._m_step(Nk, xbar, S)

    def _get_lower_bound(self):
        pass

    def _m_step(self, Nk, xbar, S):

        alpha = self.prior_params.alpha + Nk  # Bishop's equation 10.58
        beta = self.prior_params.beta + Nk  # Bishop's equation 10.60
        m = np.zeros((self.K, self.data_dim))
        v = np.zeros((1, self.K))
        invW = np.zeros((self.data_dim, self.data_dim, self.K))

        for k in range(self.K):
            if Nk[k] < 0.001:  # extinguished
                m[k, :] = self.prior_params.mean[k, :]
                invW[:, :, k] = self.prior_params.inv_s_mtx[:, :, k]
                v[:, [k]] = self.prior_params.dof[:, k]
            else:
                # Bishop's equation 10.61
                m[k, :] = (self.prior_params.beta[:, k] * self.prior_params.mean[k, :] + Nk[k] * xbar[k]) / beta[:, [k]]

                # Bishop's equation 10.62
                invW[:, :, k] = self.prior_params.inv_s_mtx[:, :, k] + (Nk[k] * S[:, :, k]) + (
                        (self.prior_params.beta[:, k] * Nk[k]) / (self.prior_params.beta[:, k] + Nk[k])) * \
                                (xbar[[k], :] - self.prior_params.mean[[k], :]).transpose().dot(
                                    xbar[[k], :] - self.prior_params.mean[[k], :])

                if np.isnan(np.sum(invW[:, :, k])):
                    print 'inverse W has NaN'

                v[:, [k]] = self.prior_params.dof[:, k] + Nk[k]  # Bishop's  equation 10.63

        return Estimate(alpha, beta, m, v, np.array([]), invW)

    def _e_step(self):
        """

        pz(i,k) = p(z=k|X(i,:), model) soft responsibility
        ll(i) = log p(X(i,:) | model)  log prob of observed data

        Calculate responsibilities using Bishop's equation 10.67

        This is the variational equivalent to the E-step in the EM algorithm.
        Here, we use the current distributions over the model parameters to evaluate
        the moments in Bishop's equations (10.64), (10.65), and (10.66), i.e., the
        expectations need to compute the responsibilities and hence evaluate E[znk] = rnk.

        """

        exp_mu_lambda, log_lambda_tilde, log_pi_tilde = self.post_params.get_expectations(self.data)

        # Bishop's equation (10.46)
        log_rho = np.tile(log_pi_tilde + 0.5 * log_lambda_tilde, (self.N, 1)) - 0.5 * exp_mu_lambda

        log_sum_rho = logsumexp(log_rho, axis=1).reshape((-1, 1))  # the reshape make the result a col-vector
        log_r = log_rho - np.tile(log_sum_rho, (1, self.K))
        r = np.exp(log_r)
        nk = np.exp(logsumexp(log_r, axis=0))
        z = log_r.max(1)  # z(i) = argmax_k p(z=k|X(i,:), model) hard clustering

        return z, r, log_sum_rho, log_r, nk

    def _compute_nk_xbar_sk(self, resp):
        """
        This computes analogous quantities to the maximum likelihood EM for the Gaussian mixture model.
        In particular, it re-estimates the parameters using the current responsibilities using Bishop's
        equation 10.51; 10.52 and 10.53.
        :param X: the input data.
        :param resp: the responsibilities for each gaussian k on the data X.
        :return:
        """

        # Bishop's equation 10.51 (Sum of responsibilities for each k)
        Nk = np.sum(resp, axis=0)

        # NUMERICAL TRICK: Adding a very small number to avoid getting 0 for a responsibility.
        Nk = Nk + 1e-10

        # responsibility weighted mean of x
        Xbar_k = np.zeros((self.K, self.data_dim))

        # responsibility weighted covariance of x
        Sk = np.zeros((self.data_dim, self.data_dim, self.K))

        for k in range(self.K):
            # Bishop's equation 10.52
            Xbar_k[k, :] = np.sum(self.data * resp[:, [k]], axis=0) / Nk[k]
            XC = self.data - Xbar_k[k, :]
            # Bishop's equation 10.53
            Sk[:, :, k] = (XC * resp[:, [k]]).conj().transpose().dot(XC) / Nk[k]

        return Nk, Xbar_k, Sk

    def _entropy(self):
        entropy = np.zeros((1, self.K))

        log_wishart_const = self.post_params.get_log_wishart_const()
        # Calculates Bishop's B.82 equation: H[Lambda]
        for k in range(self.K):
            entropy[:, k] = - log_wishart_const[:, k] - \
                            ((self.post_params.dof[:, [k]] - self.data_dim - 1) / 2) \
                            * self.post_params.log_lambda_tilde[:, k] + self.post_params.dof[:, [k]] * self.data_dim / 2
        return entropy

    def _get_lower_bound(self, Nk, xbar, S, rnk, logrnk):
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

        # getting post_params
        alpha = self.post_params.alpha
        beta = self.post_params.beta
        entropy = self._entropy()
        logDirConst = self.post_params.get_log_dir_const()
        logLambdaTilde = self.post_params.log_lambda_tilde
        logPiTilde = self.post_params.log_pi_tilde
        m = self.post_params.mean
        v = self.post_params.dof
        W = self.post_params.s_mtx

        # copying prior_params
        alpha0 = self.prior_params.alpha
        beta0 = self.prior_params.beta
        invW0 = self.prior_params.inv_s_mtx
        logDirConst0 = self.prior_params.get_log_dir_const()
        logWishartConst0 = self.prior_params.get_log_wishart_const()
        m0 = self.prior_params.mean
        v0 = self.prior_params.dof


        # Bishop's equation 10.71
        ElogpX = np.zeros((1, self.K))
        for k in range(self.K):
            xbarc = xbar[[k], :] - m[[k], :]

            # the reshape maintains the 1x2 for after the multiplications
            ElogpX[:, k] = 0.5 * Nk[k] \
                           * (logLambdaTilde[:, k] - (self.data_dim / beta[:, [k]])
                              - np.trace(v[:, [k]] * S[:, :, k].dot(W[:, :, k]))
                              - v[:, [k]] * np.sum(xbarc.dot(W[:, :, k]) * xbarc, keepdims=True)
                              - (self.data_dim * np.log(2 * np.pi)))  # Bishop's equation 10.71
        ElogpX = np.sum(ElogpX)

        # Bishop's equation 10.72
        ElogpZ = np.sum(Nk * logPiTilde)

        # Bishop's equation 10.73
        Elogppi = logDirConst0 + np.sum((alpha0 - 1) * logPiTilde)

        # Bishop's equation 10.74
        ElogpmuSigma = np.zeros((1, self.K))
        for k in range(self.K):
            mc = m[[k], :] - m0[[k], :]

            ElogpmuSigma[:, k] = 0.5 * (self.data_dim * np.log(beta0[:, k] / (2 * np.pi))
                                        + logLambdaTilde[:, k] - ((self.data_dim * beta0[:, k]) / beta[:, [k]])
                                        - beta0[:, k] * v[:, [k]] * np.sum(mc.dot(W[:, :, k]) * mc, keepdims=True)) \
                                 + logWishartConst0[:, k] \
                                 + 0.5 * (v0[:, k] - self.data_dim - 1) * logLambdaTilde[:, k] \
                                 - 0.5 * v[:, [k]] * np.trace(invW0[:, :, k].dot(W[:, :, k]))

        ElogpmuSigma = np.sum(ElogpmuSigma)

        # Entropy terms
        # Bishop's equation 10.75
        ElogqZ = np.sum(np.sum(rnk * logrnk, axis=0, keepdims=True))

        # Bishop's equation 10.76
        Elogqpi = np.sum((alpha - 1) * logPiTilde) + logDirConst

        # Bishop's equation 10.77
        ElogqmuSigma = np.sum((1 / 2) * logLambdaTilde + (self.data_dim / 2) *
                              np.log(beta / (2 * np.pi)) - (self.data_dim / 2) - entropy)

        # Overall sum
        # Bishop's equation 10.70
        L = ElogpX + ElogpZ + Elogppi + ElogpmuSigma - ElogqZ - Elogqpi - ElogqmuSigma

        if np.isnan(L):
            print "ElogpX:\n{}".format(ElogpX)
            print "ElogpZ:\n{}".format(ElogpZ)
            print "Elogppi:\n{}".format(Elogppi)
            print "ElogpmuSigma:\n{}".format(ElogpmuSigma)
            print "ElogqZ:\n{}".format(ElogqZ)
            print "Elogqpi:\n{}".format(Elogqpi)
            print "ElogqmuSigma:\n{}".format(ElogqmuSigma)

        return L

    def fit(self, max_iter=200, thresh=1e-5):
        self._init_prior()

        # Initialization -- In practice, a better initialization procedure would be to
        # choose the cluster centres mu_k to be equal to a random subset of K data points.
        # It is also worth noting that the K-means algorithm itself is often used to
        # initialize the parameters in a Gaussian mixture model before applying the
        # EM algorithm. Bishop's book pag. 427.

        # fit a Gaussian Mixture Model with two components -- Initialization
        # clf = GaussianMixture(n_components=self.K, max_iter=30, covariance_type='full', init_params='kmeans',
        #                       tol=0.1, random_state=1)
        # clf.fit(self.data)
        # xbar = clf.means_          # centres
        # Nk = self.N*clf.weights_        # priors
        S = np.zeros((self.data_dim, self.data_dim, self.K))  # covariances
        # for k in range(clf.covariances_.shape[0]):    # reshape covariances
        #     S[:, :, k] = clf.covariances_[k]

        Nk = np.array([42.6121, 33.1998, 16.3821, 47.2440, 49.9446, 82.6175])

        xbar = np.array([[0.3665, 0.3326],
                         [-1.1101, - 0.8419],
                         [-1.1632, - 1.5650],
                         [-1.4221, - 1.3396],
                         [0.6977, 1.0827],
                         [0.8791, 0.5886]])

        S[:, :, 0] = np.array([[0.0996, 0.0742],
                               [0.0742, 0.2066]])

        S[:, :, 1] = np.array([[0.0699, 0.0065],
                               [0.0065, 0.1224]])

        S[:, :, 2] = np.array([[0.0251, 0.0104],
                               [0.0104, 0.0514]])

        S[:, :, 3] = np.array([[0.0066, 0.0025],
                               [0.0025, 0.1152]])

        S[:, :, 4] = np.array([[0.1389, 0.0600],
                               [0.0600, 0.1067]])

        S[:, :, 5] = np.array([[0.0517, 0.0113],
                               [0.0113, 0.0749]])

        self._init_posterior(Nk, xbar, S)

        # -- Main loop -- #
        iter_num = 0
        done = False
        log_lik_hist = []
        converged = False

        while not done:
            # E-step
            z, rnk, ll, log_rnk, _ = self._e_step()

            # compute responsibilities variables
            Nk, xbar, S = self._compute_nk_xbar_sk(rnk)

            # get lower bound for iteration
            log_lik_hist.append(self._get_lower_bound(Nk, xbar, S, rnk, log_rnk))

            # M-step
            self.post_params = self._m_step(Nk, xbar, S)

            # if plotFn:
            #     p = self.post_params
            #     plotFn(X, p.alpha, p.m, p.W, p.v, log_lik_hist[iter_num], iter_num)

            # converged?
            if iter_num == 0:
                converged = False
            else:
                converged = convergence_test(log_lik_hist[iter_num], log_lik_hist[iter_num - 1], thresh)

            done = converged or (iter_num > max_iter)

            if self.verbose:
                print 'Iteration #{}, loglik = {}'.format(iter_num, log_lik_hist[iter_num])

            iter_num += 1

        return self.post_params, log_lik_hist

    def plot_posterior_alpha(self):
        plt.bar(range(self.K), list(self.post_params.alpha[0,:]))
        plt.title('Posterior alpha')
        plt.xlabel('K')
        plt.show()


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

    model = VBGMM(data, K)

    # run mixGaussBayesFit
    posterior, loglikHist = model.fit()

    model.plot_posterior_alpha()

    # plot likelihood
    plt.plot(loglikHist, '-', marker='*', lw=3)
    plt.yticks(np.arange(-1100, -601, 50))
    plt.xlim([0, 100])
    plt.xlabel('iterations')
    plt.ylabel('lower bound on log marginal likelihood')
    plt.title('variational Bayes objective for GMM on old faithful data')
    plt.show()


if __name__ == '__main__':
    main()
