# Python-Machine-Learning-Models
Simple Python implementations of a selection of the most commonly (and some less so) used machine learning algorithms. Studying and implementing these models will help with the understanding of their workings, unlike the black box nature of the models found in common python packages.

### DecisionTree
A classification staple. The tree object itself is found in tree.py, however all training and tree creation is carried out in decision_tree_learning in decisionTree.py. This functions uses the ID3 algorithm to recursively build the tree.
The tree takes in a binary target vector, so for use cases where the data can be multi-valued, a function for allow one-v-rest classification has been created in decisionTree.py
##### TODO
* Create a random forest classifier

### RBM
Not popular today, but a great introduction to generative models
##### TODO
* Test
* Allow classification as well as generation
* Build on the RBM class to create a more powerful Deep Belief Net

### Regression
A couple of common linear regression techniques.
* Maximum A Posteriori
* Bayesian Regression

### Gaussian Process
A powerful regression technique. Object for full GP, as well as a distributed experts model. The distributed model allows for faster training, and training time goes from O(N^3) to O(M D^3), where M is the number of experts, N is the total data points, and D is number of data points per expert.
##### TODO
* Test
* Add other kernels - seperate kernels into different classes
* Add generalised PoE, Bayesian committee machine (BCM), and generalised BCM experts models.
* Bayesian optimisation

### Hidden Markov Model
A model for finding the latent distributions when consecutive data points can be generated from different distributions.
The HMM object can work with gaussian (model = 'gauss') or multinomial (model = 'multinomial') data.
##### TODO
* Test
* Add comments to remaining functions


### In Development

* A simple feed-forward neural network
* A simple convolutional neural network
* Support vector machine
* Support vector Regression
* PCA
* LDA
* PPCA
