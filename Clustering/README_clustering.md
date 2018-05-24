# Clustering

### Kmeans


### GMM Expectation Maximisation
To see a test example of the GMM, run test_GMM.py
To use:
1. create a GMM object with input X (shape N x d, where d is number of dimensions), K (number of clusters)
2. run GMM.findParams(n_it), n_it is number of EM iterations. This returns a dictionary with keys means_i, covars_i, mixCoeff_i, i from 1 to K

TODO:
* Add __str__ function
