from scipy import stats
import numpy as np

class GMM(object):

    def __init__(self,X,K, a_seed = None):
        '''
        Inputs:
            X | np array of data, n x k, k being dimension of data
            K | integer, number of clusters to find
            a_seed | optional float to fix random numbers
        '''

        self.X = X
        self.K = K
        self.resp = None
        self.params = self._initialise(a_seed)

    def _initialise(self, a_seed):
        '''Create initial 'guesses' for the parameters
        Returns:
        A | dict of parameters'''
        if a_seed != None:
            np.random.seed(a_seed)
        dim = self.X.shape[1]
        params = {}

        mixCoeffs = np.random.random(size=(self.K,))
        mixCoeffs /= np.sum(mixCoeffs) #so they sum to 1

        min_data = np.min(self.X, axis=0)
        max_data = np.max(self.X, axis=0)

        for i in range(1, self.K + 1):
            params['means_{}'.format(i)] = np.random.uniform(min_data, max_data)
            params['covars_{}'.format(i)] = np.random.uniform(0.,1., size=(dim,dim)) + np.eye(dim)
            params['mixCoeff_{}'.format(i)] = mixCoeffs[i-1]
        return params

    def _gauss(self, x, mean, covar):
        return stats.multivariate_normal.pdf(x, mean=mean, cov=covar)

    def _expectation(self):
        '''Expectation step of the EM algorithm
        Returns:
            E | n x K numpy array of responsibilities'''
        E = np.zeros((self.X.shape[0],self.K))
        for k in range(1,self.K+1):
            for n in range(self.X.shape[0]):
                x_n = self.X[n,:]
                sum_over_mixtures = 0.0 #normalising demoninator
                for j in range(1,self.K+1):
                    pi_j = self.params['mixCoeff_{}'.format(j)]
                    mean_j = self.params['means_{}'.format(j)]
                    covars_j = self.params['covars_{}'.format(j)]
                    gauss_nj = self._gauss(x_n, mean_j, covars_j)
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
            mean_k = (1 / N_k) * np.dot(self.X.T,self.resp[:,k-1])

            #calc terms for covar
            covar_sum = 0.0 #sum quadratic (x_n - mean) * resp_nk (over n)
            for n in range(self.X.shape[0]):
                x_n = self.X[n,:]
                diff = x_n - mean_k
                covar_sum += self.resp[n,k-1] * (np.outer(diff,diff))
            covars_k = covar_sum / N_k

            pi_k = N_k / self.X.shape[0]

            A['means_{}'.format(k)] = mean_k
            A['covars_{}'.format(k)] = covars_k
            A['mixCoeff_{}'.format(k)] = pi_k
        return A

    def findParams(self, n_it):
        for i  in range(n_it):
            self.resp = self._expectation(); # compute responsibilities, i.e., every \\gamma(z_{n,k})
            self.params = self._maximisation() # update the values for the parameters
        return self.params
