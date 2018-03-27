import numpy as np
from gp import gp
import matplotlib.pyplot as plt

class distributedGP(object):

    def __init__(self, X, y,N_experts, method='poe', beta=None):
        assert N_experts < X.shape[0], "you can't have more experts than data points"
        self.method = method
        self.N_experts = N_experts
        self.X = X
        self.y = y
        self.N = X.shape[0] #Number of training points
        self.M = int(self.N / N_experts)
        self.bcm = False
        #add code to randomly shuffle X and y first
        if method == 'poe':
            self.setBeta(np.ones((N_experts)))
        elif method == 'gpoe':
            self.setBeta(beta)
        elif method == 'bcm':
            self.bcm = True
            self.setBeta(np.ones((N_experts)))
        elif method == 'gbcm':
            self.bcm = True
            self.setBeta(beta)
        else:
            raise ValueError('Distributing method must be poe, gpoe, bcm, or gbcm')


        self.experts = []
        for i in range(N_experts):
            X_expert = X[i*self.M:(i+1)*self.M, :]
            y_expert = y[i*self.M:(i+1)*self.M, :]
            sigma2_n = np.random.random()
            sigma2_s = np.random.random()
            length_scale = np.random.uniform(0,2) #how to best initialise?
            params = {}
            params['sigma2_noise'] = 1.0
            params['sigma2_signal'] = 2.0
            params['length_scale'] = 1.0
            an_expert = gp(X_expert, y_expert, "rbf", params)
            # OPTIMIZE HYPERPARAMS
            self.experts.append(an_expert)

    def setBeta(self, beta):
        if beta is None:
            beta = np.ones((self.N_experts)) * (1/float(self.N_experts))
        self.beta = beta

    def predict(self, X_test):
        """Consult the experts to predict means and covariances at test points.
        For each test point, find a 1D mean and variance.
        """
        N_test = X_test.shape[0]
        means = np.zeros((N_test,1))

        var_inv = np.zeros((N_test,1))
        for i, expert in enumerate(self.experts):
            mean_expert, var_expert = expert.predict(X_test)
            var_expert_inv = np.reciprocal(var_expert)
            var_inv += self.beta[i] * var_expert_inv
            means += self.beta[i] * var_expert_inv * mean_expert

        if self.bcm:
            an_expert = self.experts[0]
            prior_vars = np.diag(an_expert.kernel.rbf(X_test,X_test))
            prior_vars = np.reshape(prior_vars, (prior_vars.shape[0],1))
            var_inv += (1-self.M) * np.reciprocal(prior_vars)

        total_variance = np.reciprocal(var_inv)
        total_mean = total_variance * means

        return total_mean, total_variance

    def predict_gbcm(self, X_test):
        N_test = X_test.shape[0]
        means = np.zeros((N_test,1))

        var_inv = np.zeros((N_test,1))
        for i, expert in enumerate(self.experts):
            mean_expert, var_expert = expert.predict(X_test)
            var_expert_inv = np.reciprocal(var_expert)

            prior_vars = np.diag(expert.kernel.rbf(X_test,X_test))
            prior_vars = np.reshape(prior_vars, (prior_vars.shape[0],1))

            var_inv += self.beta[i] * var_expert_inv -self.beta[i] * prior_vars
            means += self.beta[i] * var_expert_inv * mean_expert
        var_inv += prior_vars #to make it (1- sum of betas)
        total_variance = np.reciprocal(var_inv)
        total_mean = total_variance * means

        return total_mean, total_variance
