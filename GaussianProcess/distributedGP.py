import numpy as np
from gp import gp

class distributedGP(object):

    def __init__(self, N_experts, X, y, method='poe'):
        self.method = method
        self.N_experts = N_experts
        self.X = X
        self.y = y
        self.N = X.shape[0] #Number of training points
        self.M = int(self.N / N_experts)

        assert(N_experts < self.N, "you can't have more experts than data points")

        #add code to randomly shuffle X and y first
        if method == 'poe':
            self.predict = self.predict_poe
        elif method == 'gpoe':
            pass
        elif method == 'bcm':
            pass
        elif method == 'gbcm':
            pass
        else:
            raise ValueError('Distributing method must be poe, gpoe, bcm, or gbcm')


        self.experts = []
        for i in range(N_experts):
            X_expert = X[i*M:(i+1)*M, :]
            y_expert = y[i*M:(i+1)*M, :]
            sigma2_n = np.random.random()
            sigma2_s = np.random.random()
            length_scale = np.random.uniform(0,2) #how to best initialise?
            params = {}
            params['sigma2_noise'] = sigma2_n
            params['sigma2_signal'] = sigma2_s
            params['length_scale'] = length_scale
            an_expert = gp(X_expert, y_expert, "rbf", params)
            # OPTIMIZE HYPERPARAMS
            self.experts.append(an_expert)

    def predictions(self, X_test):
        """Consult the experts to predict means and covariances at test points.
        For each test point, find a 1D mean and variance.
        """
        N_test = X_test.shape[0]
        means = np.zeros((N_test))
        variances = np.zeros((N_test))
        for i in range(N_test):
            data = X_test[i,:]
            mean, var = self.predict(data)
            means[i] = mean
            variances[i] = var
        return means, variances

    def predict_poe(self,X_test):
        """Given a single test point, consult the experts to
        find a 1D mean and variance.
        """
        variance_inv = 0.
        mean = 0.
        for i,expert in enumerate(self.experts):
            mean, cov = expert.predict(X_test)
            variance_inv += 1/float(cov)
            mean += cov * mean
        total_variance = 1 / float(variance_inv)
        total_mean = variance_inv * mean

        return total_mean, total_variance
