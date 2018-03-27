import numpy as np
from gp import gp
import matplotlib.pyplot as plt

class distributedGP(object):

    def __init__(self, N_experts, X, y, method='poe', beta=None):
        assert N_experts < X.shape[0], "you can't have more experts than data points"
        self.method = method
        self.N_experts = N_experts
        self.X = X
        self.y = y
        self.N = X.shape[0] #Number of training points
        self.M = int(self.N / N_experts)
        #add code to randomly shuffle X and y first
        if method == 'poe':
            self.predict_method = self.predict_poe
            self.setBeta(np.ones((N_experts)))
        elif method == 'gpoe':
            self.predict_method = self.predict_poe
            self.setBeta(beta)
        elif method == 'bcm':
            pass
        elif method == 'gbcm':
            pass
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
        total_variance = np.reciprocal(var_inv)
        total_mean = total_variance * means
        return total_mean, total_variance




params = {}
params['sigma2_noise'] = 1.0
params['sigma2_signal'] = 2.0
params['length_scale'] = 1.0

np.random.seed(42)
x = np.random.random((20,1)) * 10
y = np.cos(x) + 0.5*x + np.random.normal(loc=0.0,scale=0.2, size=(20,1))

x_test = np.linspace(-5,15,100)
x_test = np.reshape(x_test, (100,1))

a = gp(x, y, "rbf", params)
b = distributedGP(8,x,y)
c = distributedGP(8,x,y, method='gpoe')

mean, cov = a.predict(x_test)
mean2, cov2 = b.predict(x_test)
mean3, cov3 = c.predict(x_test)

plt.scatter(x,y)
plt.plot(x_test, mean)
plt.plot(x_test, mean + np.sqrt(cov)*2, linestyle='--', color='g')
plt.plot(x_test, mean - np.sqrt(cov)*2, linestyle='--', color='g')
plt.show()
