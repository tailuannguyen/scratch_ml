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
        best_silhouette = -1
        best_clustering = None
        best_centroids = None
        for _ in range(5):  # Run the algorithm multiple times to avoid poor local minima
            self.initialize_centroids(X)
            iteration = 0
            clustering = np.zeros(X.shape[0])
            while iteration < self.max_iter:
                # your code
                old_clustering = clustering.copy()
                clustering = np.argmin(self.euclidean_distance(X, self.centroids), axis=1)
                self.update_centroids(clustering, X)
                if np.array_equal(old_clustering, clustering):
                    break
                iteration += 1
            silhouette_score = self.silhouette(clustering, X)

            # store the best clustering found so far
            if silhouette_score > best_silhouette:
                best_silhouette = silhouette_score
                best_clustering = clustering
                best_centroids = self.centroids.copy()

        self.centroids = best_centroids

        return best_clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        #your code
        self.centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[clustering == k]
            if np.shape(cluster_points)[0] > 0:
                self.centroids[k] = cluster_points.mean(axis=0)
            else:
                self.centroids[k] = X[np.random.choice(a=X.shape[0], size=1, replace=False)]

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
            self.centroids = np.zeros((self.n_clusters, X.shape[1])) # Pre-allocated memory for centroids to avoid repeated memory allocation (growing arrays)
            random_indice = np.random.choice(a=X.shape[0], size=1, replace=False)
            self.centroids[0] = X[random_indice]
            for k in range(1, self.n_clusters):              
                distances = self.euclidean_distance(X, self.centroids[:k])
                min_distances = np.min(distances, axis=1)
                centroid_probs = min_distances / np.sum(min_distances)
                next_centroid_index = np.random.choice(a=X.shape[0], size=1, replace=False, p=centroid_probs)
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
        padded_X1 = X1[:, np.newaxis, :]  
        padded_X2 = X2[np.newaxis, :, :]
        square_diffs = (padded_X1 - padded_X2) ** 2 
        dists = np.sqrt(square_diffs.sum(axis=2))
        return dists

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code
        n_samples = X.shape[0]
        distances = self.euclidean_distance(X, X)
        silhouette_scores = np.zeros(n_samples)

        for i in range(n_samples):  
            same_cluster_mask = clustering == clustering[i]
            same_cluster_mask[i] = False  # Exclude self-distance to not interfere with mean calculation below
            
            # Calculate a(i)
            if np.sum(same_cluster_mask) > 0:
                a_i = distances[i, same_cluster_mask].mean()
            else:
                a_i = 0
            
            # Calculate b(i)
            b_i = np.inf
            for k in range(self.n_clusters):
                if k == clustering[i]:  # Skip its own cluster
                    continue
                cluster_k_mask = clustering == k
                if np.sum(cluster_k_mask) > 0:
                    temp_b_i = distances[i, cluster_k_mask].mean()
                    b_i = min(b_i, temp_b_i)
            
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

        return silhouette_scores.mean()
            