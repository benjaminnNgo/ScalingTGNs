import math
from sklearn.cluster import KMeans
import warnings

from sklearn.metrics import silhouette_score

# To ignore all warnings
warnings.filterwarnings("ignore")

def find_optimal_k (data,patience = 5, tolerant = 0.2):
    k = math.floor(len(data) * 0.1)
    optimal_k = k
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    best_score = silhouette_avg
    fitting = True
    stopping = 0
    while fitting and k < len(data):
        k+= 1
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)

        if silhouette_avg > best_score:
            best_score = silhouette_avg
            optimal_k =  k
            stopping = 0
        else:
            if best_score - silhouette_avg > tolerant:
                stopping += 1

        if stopping > patience:
            fitting = False
    return optimal_k





