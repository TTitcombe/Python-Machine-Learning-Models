import numpy as np
import matplotlib.pyplot as plt
from distributedGP import distributedGP
from gp import gp

params = {}
params['sigma2_noise'] = 1.0
params['sigma2_signal'] = 2.0
params['length_scale'] = 1.0

#training data
np.random.seed(42)
x = np.random.random((20,1)) * 10
y = np.cos(x) + 0.5*x + np.random.normal(loc=0.0,scale=0.2, size=(20,1))

#test data X points
x_test = np.linspace(-5,15,100)
x_test = np.reshape(x_test, (100,1))


a_gp = gp(x, y, "rbf", params)
product_of_experts = distributedGP(x,y,8)
general_poe = distributedGP(x,y,8, method='gpoe')
bcm = distributedGP(x,y,8, method='bcm')
gbcm = distributedGP(x,y,8,method='gbcm')

mean, cov = a_gp.predict(x_test)
mean2, cov2 = product_of_experts.predict(x_test)
mean3, cov3 = general_poe.predict(x_test)
mean4, cov4 = bcm.predict(x_test)
mean5, cov5 = gbcm.predict_gbcm(x_test)

plt.scatter(x,y)
plt.plot(x_test, mean5)
plt.plot(x_test, mean5 + np.sqrt(cov5)*2, linestyle='--', color='g')
plt.plot(x_test, mean5 - np.sqrt(cov5)*2, linestyle='--', color='g')
plt.show()
