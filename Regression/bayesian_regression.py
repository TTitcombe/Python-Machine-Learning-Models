import numpy as np
import matplotlib.pyplot as plt


def calc_gauss_phi(k,x):
    '''This function calculates the basis
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

def calc_gauss_cphi(k,x):
    '''This function calculates the weights matrix.
    Parameters:
        k: scalar | order of basis functions
        x: float, matrix | matrix of data  
    Returns:
        phi: float, matrix | weights matrix
    '''
    phi = np.zeros((len(x),k+1))
    for i in range(len(x)):
        vals = calc_gauss_phi(k,x[i])
        phi[i,:] = vals
    return phi
    
def posterior(Phi, var, S0, m0, Y):
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
    
#------Hyperparameters--------#
alpha = 1.0
beta = 0.1


#----Generate Data---------#
N = 25
X = np.reshape(np.linspace(0,0.9,N), (N,1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
X_test = np.reshape(np.linspace(-1.0,1.5,200),(200,1))
Y_test = np.cos(10*X_test**2) + 0.1 + np.sin(100*X_test)

#---Generate Phi Matrix--------------#
Phi_test = calc_gauss_cphi(10, X_test)
Phi = calc_gauss_cphi(10, X)

#----Prior on weights----------#
m0 = np.zeros((11))
S0 = alpha * np.eye(11)

#----Calculate posterior------#
Sn, mn = posterior(Phi, beta, S0, m0, Y)

#----Calculate noise-free mean and variance----#
NFMean = np.dot(mn.T, Phi_test.T)
NFvar = np.diagonal(np.dot(Phi_test, np.dot(Sn, Phi_test.T)))

#---Take N_samples from the weights posterior----#
N_samples = 5
theta_int = np.random.multivariate_normal(mn.T[0], Sn, N_samples)
test_vals = np.dot(Phi_test, theta_int.T)

for i in range(N_samples):
    plt.plot(X_test, test_vals[:,i],linewidth=5,zorder=i, label='Sample: {}'.format(i+1))
    
plt.fill_between(X_test.T[0], NFMean[0]+2*np.sqrt(NFvar),
        NFMean[0]-2*np.sqrt(NFvar),color='palegreen',alpha=0.6, label='Noise Free')
plt.plot(X_test, NFMean[0],linewidth=5, label='Predictive Mean')
plt.scatter(X,Y, s=100,c='k',label='Training Data', zorder=10)
plt.legend(loc='upper right', fontsize=14)
plt.xlim([-1.0,1.5])
plt.ylim([-2.8,3.9])
plt.xlabel('X', fontsize=32)
plt.ylabel('Y', fontsize=32)

plt.show()