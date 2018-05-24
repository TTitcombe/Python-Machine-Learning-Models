import numpy as np
import matplotlib.pyplot as plt

from GMM import GMM

X1 = np.random.uniform([-1.,1.], [0., 5], size=(20,2))
X2 = np.random.uniform([0, 6.], [2., 10.], size=(30,2))

X = np.vstack((X1, X2))
np.random.shuffle(X)

gmm = GMM(X, 2)
params = gmm.findParams(50)


plt.scatter(X1[:,0], X1[:,1], color='r')
plt.scatter(X2[:,0], X2[:,1], color='b')
for i in range(1,3):
    mean = params['means_{}'.format(i)]
    plt.scatter(mean[0], mean[1], marker='x', color='g')
    mix = params['mixCoeff_{}'.format(i)]
    print(mix)

plt.show()
