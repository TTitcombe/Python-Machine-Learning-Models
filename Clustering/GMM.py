from scipy import stats
import numpy as np

class GMM(object):

    def __init__(self,X,K,dim=2, a_seed = None):
        assert type(dim) == int, "Dimension of data must be an integer"

        self.X = X
        self.K = K
        self.resp = None
        self.dim = dim
        self.params = self.initialise(dim, a_seed)

    def initialise(self,dim, a_seed):
        '''Create initial 'guesses' for the parameters
        Returns:
        A | dict of parameters'''
        if a_seed != None:
            np.random.seed(a_seed)
        params = {}
        for i in range(1, self.K + 1):
            params['means_{}'.format(i)] = np.random.rand(dim)
            params['covars_{}'.format(i)] = np.random.rand(dim,dim) + np.eye(dim)
            params['mixCoeff_{}'.format(i)] = np.random.rand(1)
        return params

    def gauss(self, x, mean, covar):
        return stats.multivariate_normal.pdf(x, mean=mean, cov=covar)

    def _expectation(self):
        '''Expectation step of the EM algorithm
        Returns:
            E | n x K numpy array of responsibilities'''
        E = np.zeros((self.X.shape[1],self.K))
        for k in range(1,self.K+1):
            for n in range(self.X.shape[1]):
                x_n = self.X[:,n]
                sum_over_mixtures = 0.0 #normalising demoninator
                for j in range(1,self.K+1):
                    pi_j = self.params['mixCoeff_{}'.format(j)]
                    mean_j = self.params['means_{}'.format(j)]
                    covars_j = self.params['covars_{}'.format(j)]
                    gauss_nj = self.gauss(x_n, mean_j, covars_j)
                    if j == k:
                        numerator = gauss_nj * pi_j
                    sum_over_mixtures += gauss_nj * pi_j
                gamma_nk = numerator / float(sum_over_mixtures)
                E[n,k-1] = gamma_nk
        return E

    def _maximisation(self):
        '''Maximising step of the EM algorithm
        Returns:
            A | dictionary of params'''
        A = {}
        for k in range(1,self.K+1):
            mean_sum = 0.0 #sum of responsibilities nk times data x_n (over n)
            N_k = sum(self.resp[:,k-1])
            mean_k = (1 / N_k) * np.dot(self.X,self.resp[:,k-1])

            #calc terms for covar
            covar_sum = 0.0 #sum quadratic (x_n - mean) * resp_nk (over n)
            for n in range(self.X.shape[1]):
                x_n = self.X[:,n]
                diff = x_n - mean_k
                covar_sum += self.resp[n,k-1] * (np.outer(diff,diff))
            covars_k = covar_sum / N_k

            pi_k = N_k / self.X.shape[1]

            A['means_{}'.format(k)] = mean_k
            A['covars_{}'.format(k)] = covars_k
            A['mixCoeff_{}'.format(k)] = pi_k
        return A

    def findParams(self, n_it):
        for i  in range(n_it):
            self.resp = self._expectation(); # compute responsibilities, i.e., every \\gamma(z_{n,k})
            self.params = self._maximisation() # update the values for the parameters
        return params
