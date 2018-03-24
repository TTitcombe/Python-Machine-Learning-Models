"""A Gaussian Process object.

This was completed as part of coursework for the Imperial College London
course 'Data Analysis and Probabilistic Inference'.

Wrapper, and optimize functions, created by Marc Deisenroth."""
import numpy as np
from scipy.optimize import minimize

class gp(object):

    def __init__(self, X, y, kernel, params):
        if kernel == "rbf":
            self.kernel_func = self.rbf
        self.X = X
        self.y = y
        self.N = X.shape[0]
        self.K = self.calc_kernel(X)

        self.setParams(params)

    def multivariateGaussianDraw(self, mean, cov):
        normalised_sample = np.random.normal(size=(mean.shape[0],))
        L = np.linalg.cholesky(cov)
        sample = np.dot(L, normalised_sample) + mean
        return sample

    def setParams(self, params):
        self.sigma2_noise = params.get("sigma2_noise", 0.1)
        self.sigma2_signal = params.get("sigma2_signal", 0.1)
        self.length_scale = params.get("length_scale", 1.)

        self.ln_noise = np.log(self.sigma2_noise)
        self.ln_signal = np.log(self.sigma2_signal)
        self.ln_length = np.log(self.length_scale)

    def getParams(self):
        params = {}
        params['sigma2_noise'] = self.sigma2_noise
        params['sigma2_signal'] = self.sigma2_signal
        params['length_scale'] = self.length_scale
        return params

    def getParams_log(self):
        params = {}
        params['ln_noise'] = self.ln_noise
        params['ln_signal'] = self.ln_signal
        params['ln_length'] = self.ln_length
        return params

    def rbf(self,xi, xj):
        diff = xi - xj
        return self.sigma2_signal * np.exp(-np.dot(diff, diff.T) / (2.0*self.length_scale**2))

    def calc_kernel(self, X, X_2 = None, params = None):
        if params is not None:
            self.setParams(params)
        if X_2 is not None:
            X = np.vstack(X,X_2)
        N = X.shape[0]
        K = np.zeros((N,N))
        for i in range(N):
            xi = X[i,:]
            for j in range(N):
                xj = X[j,:]
                K[i,j] = self.kernel_func(xi,xj)
        return K + self.sigma2_noise * np.identity(N)

    def predict(self, X_2):
        post_mean = np.zeros((X_2.shape[0], 1))
        post_cov = np.zeros((X_2.shape[0], X_2.shape[0]))

        Sigma = self.calc_kernel(self.X, X_2)
        K_dim = self.K.shape[0] #N in X train
        k_train_test = Sigma[0:K_dim, K_dim:]
        k_test_train = k_train_test.T
        k_test_test = Sigma[K_dim:, K_dim:]

        #Posterior Mean calculation
        kalman_gain = np.linalg.inv(self.K) #K already has noise term
        kalman_gain = np.dot(k_test_train, kalman_gain)
        post_mean = np.dot(kalman_gain, self.y)

        #Posterior covariance calculation
        post_cov_term_1 = k_test_test - self.k.sigma2_noise * np.identity(k_test_test.shape[0])
        post_cov = post_cov_term_1 - np.dot(kalman_gain, k_train_test)

        return post_mean, post_cov

    def logMarginalLikelihood(self, params=None):
        if params is not None:
            self.K = self.calc_kernel(self.X, params)
        K = self.K
        N = K.shape[0]
        mll = 0.
        K_inv = np.linalg.inv(K)

        term_1 = np.dot(self.y.T, np.dot(K_inv, self.y))
        sign, logdet = np.linalg.slodget(K)
        term_2 = sign * logdet
        term_3 = n * np.log(2.0 * np.pi)

        mll = 0.5 * (term_1 + term_2 + term_3)
        return mll

    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            self.K = self.calc_kernel(self.X, params)
        K = self.K
        sigma2_n = self.sigma2_noise
        sigma2_s = self.sigma2_signal
        l = self.length_scale

        l_negative2 = np.power(l,-2)
        n_K = K.shape[0]
        K_noNoise = K - sigma2_n * np.identity(n_K)
        K_inv = np.linalg.inv(K)

        alpha = np.dot(K_inv, self.y)
        term_1 = np.dot(alpha, alpha.T) - K_inv

        dK_dln_sigma_s = 2.0 * K_noNoise
        dK_dln_sigma_n = 2.0 * sigma2_n * np.identity(n_K)

        xDiffs = np.zeros((self.n, self.n))
        for i in range(self.n):
            data_i = self.X[i,:]
            for j in range(self.n):
                data_j = self.X[j,:]
                xDiff = data_i - data_j
                xDiffs[i,j] = np.dot(xDiff, xDiff.T)
        dK_dln_l = K_noNoise * xDiffs * l_negative2

        sigma_s_term = np.dot(term_1, dK_dln_sigma_s)
        sigma_n_term = np.dot(term_1, dK_dln_sigma_n)
        l_term = np.dot(term_1, dK_dln_l)

        grad_ln_s = -0.5 * np.trace(sigma_s_term)
        grad_ln_n = -0.5 * np.trace(sigma_n_term)
        grad_ln_l = -0.5 * np.trace(l_term)

        gradients = np.array([grad_ln_s, grad_ln_l, grad_ln_n])

        return gradients

        def mse(self, ya, fbar):
            n = float(ya.shape[0])
            mse = float(np.sum((ya - fbar)**2)) / n
            return mse

        def msll(self, ya, fbar, cov):
            msll = 0
            n = ya.shape[0]
            for i in range(n):
                sigma2 = cov[i,i] + self.sigma2_noise
                msll += 0.5 * np.log(2.0*np.pi * sigma2) + (ya[i] - fbar[i])**2 / (2.0 * sigma2)

            msll = msll / float(n)
            return msll

        def optimize(self, params, disp=True):
            res = minimize(self.logMarginalLikelihood, params,  method='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
            return res.X
