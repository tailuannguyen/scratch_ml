import numpy as np


class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        while iteration < self.max_iter:
            # your code
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        #your code

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            # your code
            random_indices = np.random.choice(a=X.shape[0], size=self.n_clusters, replace=False)
            self.centroids = X[random_indices]
        elif self.init == 'kmeans++':
            # your code
            self.centroids = np.zeros((self.n_clusters, X.shape[1]))
            # Initialize the first centroid randomly
            first_centroid_index = np.random.choice(a=X.shape[0], size=1, replace=False)
            self.centroids[0] = X[first_centroid_index]
            # Initialized the remaining centroids
            for k in range(1, self.n_clusters):              
                distances = self.euclidean_distance(X, self.centroids[:k])
                min_distances = np.min(distances, axis=1)
                next_centroid_probs = min_distances / np.sum(min_distances)
                next_centroid_index = np.random.choice(a=X.shape[0], size=1, replace=False, p=next_centroid_probs)
                self.centroids[k] = X[next_centroid_index]
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        dists = np.sqrt(((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2).sum(axis=2))
        return dists

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code