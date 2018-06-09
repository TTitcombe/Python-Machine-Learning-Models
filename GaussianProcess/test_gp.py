import numpy as np
import matplotlib.pyplot as plt
from distributedGP import distributedGP
from gp import GP

params = {}
params['ln_noise'] = 0.5
params['ln_signal'] = 1.0
params['ln_length'] = 0.3

#training data
np.random.seed(42)
x = np.random.random((50,1)) * 20
y = np.cos(x) + 0.5*x + np.random.normal(loc=0.0,scale=0.2, size=(50,1))

#test data X points
x_test = np.linspace(-5,22,100)
x_test = np.reshape(x_test, (100,1))


a_gp = GP(x, y, "matern", params)
b = GP(x,y,"rbf",params)
'''
product_of_experts = distributedGP(x,y,8)
general_poe = distributedGP(x,y,8, method='gpoe')
bcm = distributedGP(x,y,8, method='bcm')
rbcm = distributedGP(x,y,8,method='rbcm')
'''

mean, cov = a_gp.predict(x_test)
m2,c2 = b.predict(x_test)
'''
mean2, cov2 = product_of_experts.predict(x_test)
mean3, cov3 = general_poe.predict(x_test)
mean4, cov4 = bcm.predict(x_test)
mean5, cov5 = rbcm.predict(x_test)
'''

plt.scatter(x,y)
plt.plot(x_test, mean,c='g', label='matern mean')
plt.plot(x_test, mean + np.sqrt(cov)*2, linestyle='--', color='g', label='matern var')
plt.plot(x_test, mean - np.sqrt(cov)*2, linestyle='--', color='g')
plt.plot(x_test,m2,c='orange', label='rbf mean')
plt.plot(x_test, m2 + np.sqrt(c2)*2, c='orange',linestyle='--', label='rbf var')
plt.plot(x_test, m2 - np.sqrt(c2)*2, c='orange', linestyle='--')
plt.xlabel('X')
plt.ylabel('y vals')
plt.legend()
plt.show()
