import numpy as np

class bayesian_regression(object):
    
    def __init__(self,alpha, beta):
        self.beta = beta #prior belief on weights covariance (assuming isotropic)
        self.mn = None #posterior mean
        self.Sn = None #posterior Covariance matrix
    
    def calc_gauss_phi(self,k,x):
        '''Calculates the basis
        function vector for a given input value.
        Parameters:
            k: scalar | order of basis functions
            x: float | a single data point
        Returns:
            R: float, vector | basis function vector evaluated at x
        '''
        R = np.ones((k+1))
        means = np.linspace(-0.5,1,k)
        scale = 0.1
        for i in range(1,k+1):
            R[i] = np.exp(-(x-means[i-1])**2 / (2.0*scale**2))
        return R
    
    def calc_gauss_cphi(self,k,x):
        '''Calculates the weights matrix.
        Parameters:
            k: scalar | order of basis functions
            x: float, matrix | matrix of data  
        Returns:
            phi: float, matrix | weights matrix
        '''
        phi = np.zeros((len(x),k+1))
        for i in range(len(x)):
            vals = self.calc_gauss_phi(k,x[i])
            phi[i,:] = vals
        return phi
        
    def posterior(self,Phi, var, S0, m0, Y):
        '''Calculates posterior distribution over the weights.
        Parameters:
            Phi: float, matrix | weight matrix
            var: float | hyperparameter 
            S0: float, matrix | prior covariance matrix on the weights
            m0: float, vector | prior mean on the weights
            Y: float, vector | training Y values
        Returns:
            Sn: float, matrix | posterior covariance matrix on the weights
            mn: float, vector | posterior mean on the weights
        '''
        Sn = np.linalg.inv(np.linalg.inv(S0) + np.dot(Phi.T,Phi)/var)
        S0m0 = np.reshape(np.dot(np.linalg.inv(S0),m0), (11,1))
        PhiY = np.dot(Phi.T,Y) / var
        mn = np.dot(Sn, (S0m0 + PhiY))
        return Sn, mn
        
    def train(self, order, X, Y, m0, S0):
        print('Beginning training....')
        Phi = self.calc_gauss_cphi(order, X)
        Sn, mn = self.posterior(Phi, self.beta, S0, m0, Y)
        self.Sn = Sn
        self.mn = mn
        print('Training done.')
        
    def predict(self,order, X_test):
        Phi_test = self.calc_gauss_cphi(order, X_test)
        if self.mn == None or self.Sn == None:
            return 'Need to train first!'
        NFMean = np.dot(self.mn.T, Phi_test.T) #noise free mean
        NFVar = np.diagonal(np.dot(Phi_test, np.dot(self.Sn, Phi_test.T)))
        return NFMean[0], NFVar
        
    def sample(self, n_samples, Phi_test):
        if self.mn == None:
            return 'Need to train first!'
        theta_int = np.random.multivariate_normal(self.mn.T[0], self.Sn, n_samples)
        test_vals = np.dot(Phi_test, theta_int.T)
        return test_vals
        
    def score(self,y_hat,y):
        return np.sum((y_hat-y)**2) / len(y)

    #still to do: create an function to plot marginal likelihood
    #for easy model selection