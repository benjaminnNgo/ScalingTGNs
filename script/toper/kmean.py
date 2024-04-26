import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings

from sklearn.metrics import silhouette_score

# To ignore all warnings
warnings.filterwarnings("ignore")

# Read the CSV file into a pandas DataFrame
data = pd.read_csv("./toper_values/data_lt_25MB/ARC0xc82e3db60a52cf7529253b4ec688f631aad9e7c2_normalization.csv")

X = []
for index, row in data.iterrows():
    point = []
    point.append(row['x'])
    point.append(row['y'])
    X.append(point)

# print(X)
X = np.array(X)



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
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)

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








# optimal_k = find_optimal_k(X)
# print(optimal_k)
# kmeans = KMeans(n_clusters=60, random_state=42)
# data['cluster'] = kmeans.fit_predict(X)
#
# # Plot all points colored by cluster
# plt.figure(figsize=(10, 6))
# # plt.scatter(data['x'], data['y'], c=data['cluster'], cmap='viridis')
# # print(kmeans.cluster_centers_)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=400, c='red', label='Centroids')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('KMeans Clustering')
# plt.legend()
# plt.show()
#
# silhouette_scores = []
#
# # Iterate over a range of cluster numbers and calculate silhouette score for each k
# for k in range(40, 100):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans.fit_predict(X)
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     silhouette_scores.append(silhouette_avg)
#
# # Plot the silhouette scores for different values of k
# plt.figure(figsize=(10, 6))
# plt.plot(range(40, 100), silhouette_scores, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score for KMeans Clustering')
# plt.xticks(range(2, 11))
# plt.show()

# dataset_per_pack = {
#
#     "data_gt_70": [
#         'AKITA0x3301ee63fb29f863f2333bd4466acb46cd8323e6',
#         'BNB0xb8c77482e45f1f44de1745f52c74426c631bdd52',
#         'CRO0xa0b73e1ff0b80914ab6fe0444e65848c4c34450b',
#         'Geminidollar0x056fd409e1d7a124bd7017459dfea2f387b6d5cd',
#         'CVC0x41e5560054824ea6b0732e656e3ad64e20e94e45'
#     ],
#     "data_lt_25MB": [
#         'ARC0xc82e3db60a52cf7529253b4ec688f631aad9e7c2',
#         'FNKOS0xeb021dd3e42dc6fdb6cde54d0c4a09f82a6bca29',
#         'INU0xc76d53f988820fe70e01eccb0248b312c2f1c7ca',
#         'unnamedtoken270x00000000051b48047be6dc0ada6de5c3de86a588',
#         'unnamedtoken124310x04906695d6d12cf5459975d7c3c03356e4ccd460'
#     ],
#
#     "data_bw_25_and_40": [
#         'CMT0xf85feea2fdd81d51177f6b8f35f0e6734ce45f5f',
#         'CELR0x4f9254c83eb525f9fcf346490bbb3ed28a81c667',
#         'GHST0x3f382dbd960e3a9bbceae22651e88158d2791550',
#         'REP0xe94327d07fc17907b4db788e5adf2ed424addff6',
#         'RFD0x955d5c14c8d4944da1ea7836bd44d54a8ec35ba1'
#     ],
# }