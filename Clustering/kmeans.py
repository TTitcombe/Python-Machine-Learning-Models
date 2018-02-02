import numpy as np

class kmeans(object):
    def __init__(self):
        self.centres = None

    def __str__(self):
        s = "A k-means clustering algorithm object. Functions are: \n"
        s += "train(X,K,n_it,a_seed) | calculates and returns the cluster centres \n"
        s += "getClusterCentres() | return cluster centres \n"
        return s

    def train(self,X, K, n_it=10, a_seed=None):
        '''A k-means clustering algorithm.
        Inputs:
            X | n x F numpy array
            K | number of clusters
            n_it | number of iterations
        Outputs: 
            cluster_centres | F x K numpy matrix of cluster means'''
        if a_seed != None:
            np.random.seed(a_seed)

        random_centres = np.random.randint(0,X.shape[1],K)
        cluster_centres = X[:,random_centres]
    
        for i in range(n_it):
            #best_distances is distance of each data point from its nearest cluster
            best_distances = np.ones((X.shape[1]))*10000
            best_distances_cluster = np.zeros((X.shape[1]))
            for cluster in range(K):
                diff = np.zeros((X.shape[0],X.shape[1])) 
                for j in range(X.shape[1]):
                    #find distance of each data point from the mean
                    diff[:,j] = X[:,j] - cluster_centres[:,cluster]
                dist = np.sqrt(diff[0,:]**2 + diff[1,:]**2) #euclidean distance
                best_distances_cluster[dist < best_distances] = cluster
                best_distances = np.minimum(best_distances, dist)
            for cluster in range(K):
                cluster_data = X[:,best_distances_cluster == cluster]
                cluster_mean = np.mean(cluster_data, axis=1) #mean along the columns
                cluster_centres[:,cluster] = cluster_mean
        self.centres = cluster_centres
        return self.centres

    def getClusterCentres(self):
        return self.centres


