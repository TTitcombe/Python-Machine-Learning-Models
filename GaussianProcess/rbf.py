import numpy as np

class rbf(object):

    def __init__(self, X, params):
        self.setParams(params)

    def setParams(self, ln_params):
        self.ln_noise = ln_params['ln_noise']
        self.ln_signal = ln_params['ln_signal']
        self.ln_length = ln_params['ln_length']

        self.sigma2_noise = np.exp(2.0*ln_params['ln_noise'])
        self.sigma2_signal = np.exp(2.0*ln_params['ln_signal'])
        self.length_scale = np.exp(ln_params['ln_length'])

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

    def kernel_func(self,xi, xj):
        diff = xi - xj
        return self.sigma2_signal * np.exp(-np.dot(diff, diff.T) / (2.0*self.length_scale**2))

    def calc_kernel(self, X, X_2 = None, params = None):
        if params is not None:
            self.setParams(params)
        if X_2 is not None:
            X_aug = np.zeros((X.shape[0]+X_2.shape[0], X.shape[1]))
            X_aug[:X.shape[0],:] = X
            X_aug[X.shape[0]:,:] = X_2
            X = X_aug
        N = X.shape[0]
        K = np.zeros((N,N))
        for i in range(N):
            xi = X[i,:]
            for j in range(N):
                xj = X[j,:]
                K[i,j] = self.kernel_func(xi,xj)
        if self.sigma2_noise is not None:
            K = K + self.sigma2_noise * np.identity(N)
        return K
