import numpy as np
from gp import gp
import matplotlib.pyplot as plt

class distributedGP(object):

    def __init__(self, N_experts, X, y, method='poe', beta=None):
        assert N_experts < self.N, "you can't have more experts than data points"
        self.method = method
        self.N_experts = N_experts
        self.X = X
        self.y = y
        self.N = X.shape[0] #Number of training points
        self.M = int(self.N / N_experts)
        #add code to randomly shuffle X and y first
        if method == 'poe':
            self.predict = self.predict_poe
            self.setBeta(np.ones((N_experts)))
        elif method == 'gpoe':
            self.predict = self.predict_poe
            self.setBeta(beta)
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

    def setBeta(self, beta):
        if beta is None:
            beta = np.ones((self.N_experts)) * (1/float(self.N_experts))
        self.beta = beta

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
        find a 1D mean and variance. Using (generalised) product of experts.
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

np.random.seed(42)
size = (100,1)
x = np.random.random(size) * 7

y = np.cos(x) + 1 + x+ np.random.normal(loc=0.0,scale=0.2, size=size)
x2 = np.random.random((10,1)) * 7
y2 = np.cos(x2) + 1 + x2 + np.random.normal(loc=0.0,scale=0.2, size=(10,1))
params = {}
params['sigma2_noise'] = 80.
params['sigma2_signal'] = 1.0
params['length_scale'] = 100.0
a = gp(x, y, "rbf", params)
mean, cov = a.predict(x2)
plt.scatter(x,y)
plt.plot(x2,mean)
plt.scatter(x2, y2)
plt.show()
print(mean)
