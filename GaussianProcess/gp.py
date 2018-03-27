"""A Gaussian Process object.

This was completed as part of coursework for the Imperial College London
course 'Data Analysis and Probabilistic Inference'.

Wrapper, and optimize functions, created by Marc Deisenroth."""
import numpy as np
from scipy.optimize import minimize
from rbf import rbf

class gp(object):

    def __init__(self, X, y, kernel, params):
        if kernel == "rbf":
            self.kernel = rbf(X,params)
        self.X = X
        self.y = y
        self.N = X.shape[0]
        self.KMat(X,params)

    def multivariateGaussianDraw(self, mean, cov):
        normalised_sample = np.random.normal(size=(mean.shape[0],))
        L = np.linalg.cholesky(cov)
        sample = np.dot(L, normalised_sample) + mean
        return sample

    def KMat(self, X, params=None):
        if params is not None:
            self.kernel.setParams(params)
        K = self.kernel.calc_kernel(X)
        self.K = K
        return K

    def train(self,X,y, params=None):
        self.X = X
        self.y = y
        self.kernel.calc_kernel(X, params=params)

    def predictPoint(self,X_test):
        Sigma = self.kernel.calc_kernel(self.X, X_test)
        K_dim = self.K.shape[0] #N in X train
        k_train_test = Sigma[:K_dim, K_dim:]
        k_test_train = k_train_test.T
        k_test_test = Sigma[K_dim:, K_dim:]

        #Posterior Mean calculation
        kalman_gain = np.linalg.inv(self.K) #K already has noise term
        kalman_gain = np.dot(k_test_train, kalman_gain)
        post_mean = np.dot(kalman_gain, self.y)

        #Posterior covariance calculation
        if self.kernel.sigma2_noise is not None:
            k_test_test = k_test_test - self.kernel.sigma2_noise
        post_cov = k_test_test - np.dot(kalman_gain, k_train_test)

        return post_mean, post_cov

    def predict(self, X_2):
        post_means = np.zeros((X_2.shape[0],1))
        post_vars = np.zeros((X_2.shape[0],1))

        for i in range(X_2.shape[0]):
            X_point = X_2[i,:]
            m, v = self.predictPoint(X_point)
            post_means[i,:] = m
            post_vars[i,:] = v

        return post_means, post_vars

    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)
        else:
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
            self.K = self.KMat(self.X, params)
        else:
            K = self.K
        sigma2_n = self.kernel.sigma2_noise
        sigma2_s = self.kernel.sigma2_signal
        l = self.kernel.length_scale

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
                sigma2 = cov[i,i] + self.kernel.sigma2_noise
                msll += 0.5 * np.log(2.0*np.pi * sigma2) + (ya[i] - fbar[i])**2 / (2.0 * sigma2)

            msll = msll / float(n)
            return msll

        def optimize(self, params, disp=True):
            res = minimize(self.logMarginalLikelihood, params,  method='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
            return res.X
