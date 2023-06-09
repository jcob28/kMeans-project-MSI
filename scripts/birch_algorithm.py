from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import Birch


class AlgBirch(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters, random_state=None):
        self.cluster_centers_ = None
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = Birch(n_clusters=n_clusters)

    def partial_fit(self, X, y, classes=None, chunk_size=1000):
        self.kmeans.partial_fit(X, y)
        self.cluster_centers_ = self.kmeans.subcluster_centers_

    def predict(self, X):
        return self.kmeans.predict(X)
