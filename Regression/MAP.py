import numpy as np
import matplotlib.pyplot as plt

#---Polynomial weight matrix-----#
def calc_poly_phi(k,x):
    '''This function calculates the polynomial basis
    function vector for a given input value.
    Parameters:
        k: scalar | order of basis functions
        x: float | a single data point
    Returns:
        R: float, vector | basis function vector evaluated at x
    '''
    R = np.zeros((k+1))
    R[0] = 1
    for i in range(1,k+1):
        R[i] = x**i
    return R
    
def calc_poly_cphi(k,x):
    '''This function calculates the polynomial phi matrix.
    Parameters:
        k: scalar | order of basis functions
        x: float, matrix | matrix of data  
    Returns:
        phi: float, matrix | values matrix
    '''
    phi = np.zeros((len(x), k+1))
    for i in range(len(x)):
        vals = calc_poly_phi(k,x[i])
        phi[i,:] = vals
    return phi

#----Trigonometric weight matrix-----#
def calc_trig_phi(k,x):
    '''This function calculates the trigonometric basis
    function vector for a given input value.
    Parameters:
        k: scalar | order of basis functions
        x: float | a single data point
    Returns:
        R: float, vector | basis function vector evaluated at x
    '''
    R = np.zeros((k*2+1))
    R[0] = 1
    v = np.pi * 2 * x
    for i in range(1,k+1):
        R[2*i-1] = np.sin(v * i)
        R[2*i] = np.cos(v * i)
    return R

def calc_trig_cphi(k,x):
    '''This function calculates the trigonometric phi matrix.
    Parameters:
        k: scalar | order of basis functions
        x: float, matrix | matrix of data  
    Returns:
        phi: float, matrix | values matrix
    '''
    phi = np.zeros((len(x),2*k+1))
    for i in range(len(x)):
        vals = calc_trig_phi(k,x[i])
        phi[i,:] = vals
    return phi

def calc_w(p,y, beta):
    '''Create the weights matrix.
    Parameters:
        p: float, matrix | Phi (x values) matrix
        y: float, vector | y values corresponding to inputs
        beta: float | parameter to avoid overfitting
                        - 0.0 is maximum likelihood
    Returns:
        w: float, matrix | parameter weights matrix
    '''
    betaEye = beta * np.eye(p.shape[1])
    return np.dot(np.linalg.inv(np.dot(p.T,p) + betaEye),np.dot(p.T,y))

def calc_y_hat(w,p):
    '''Calculate predicted y values.'''
    return np.dot(p,w)

def calc_results(basis, x_train,y_train,x,order, beta=0.0):
    '''Wrapper function.
    Parameters:
        basis: 'poly' or 'trig' | determines which basis functions to use
        x_train: float, vector | training inputs
        y_train: float, vector | training y values
        x: float, vector | test inputs
        order: scalar, array | order of basis functions to model
        beta: scalar | parameter to avoid overfitting
                        - 0.0 is max likelihood
    Returns:
        y_hat: float, vector | modeled y values at test inputs
    '''
    if basis == 'trig':
        cfunc = calc_trig_cphi
    else:
        cfunc = calc_poly_cphi
        
    #Generate weights
    phi_train = cfunc(order,x_train)
    w = calc_w(phi_train,y_train, beta)
    
    #Calculate values at test points
    phi_test = cfunc(order,x)
    y_hat = calc_y_hat(w,phi_test)
    return y_hat
    
def plots(basis, x_train,y_train,x,k, beta):
    '''Plot basis functions'''
    for order in k:
        y_hat = calc_results(basis, x_train,y_train,x,order, beta)
        plt.plot(x, y_hat,linewidth=3, label='$Order$' +' '+ str(order))
#-----------------------------------------------------------------------#
        
#---Generate data----#
N = 25
X = np.reshape(np.linspace(0,0.9,N), (N,1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
X_plot = np.reshape(np.linspace(-1.0,1.2,200),(200,1))
Y_plot = np.cos(10*X_plot**2) + 0.1*np.sin(100*X_plot)

#----Input parameters----#
beta = 1.0 #overfitting
orders = [1,3,11]

#----Plots--------#
plots('trig',X,Y,X_plot, orders, beta)
#plots('poly', X, Y, X_plot, Y_plot, orders, beta)

plt.scatter(X,Y,marker='x',color='k', s=100, label='$Training$ $Data$')
plt.legend(fontsize=24)
plt.xlabel('$X$', fontsize=30)
plt.ylim([-1.5,1.5])
plt.xlim([-1.0,1.2])
plt.ylabel('$Y$', fontsize=30)
plt.show()