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
##### TODO
* Test
* Add other kernels - seperate kernels into different classes
* Bayesian optimisation


### In Development

* A simple feed-forward neural network
* A simple convolutional neural network
* Support vector machine
* Support vector Regression
* PCA
* LDA
* PPCA
