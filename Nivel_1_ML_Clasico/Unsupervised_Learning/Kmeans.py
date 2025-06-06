# ---------- IMPORTS --------------
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# --- LOAD DATA ---
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.7, random_state=42)

# Plot the data points using a colormap to distinguish clusters
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title("Data generated with make_blobs")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

kmeans_model = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans_model.fit(X)

# Cluster labels assigned to each data point
y_kmeans_pred = kmeans_model.labels_
print(y_kmeans_pred)
# Coordinates of the cluster centroids
centroides_kmeans = kmeans_model.cluster_centers_
print(centroides_kmeans)
# Sum of squared distances of samples to their closest cluster center
inertia_kmeans = kmeans_model.inertia_
print(inertia_kmeans)

# Plotting K-Means clustering results with cluster centroids
plt.scatter(X[:,0], X[:,1], c=y_kmeans_pred, cmap='viridis')
plt.scatter(centroides_kmeans[:,0], centroides_kmeans[:,1], s=150, c='red', marker='o', label='Centroids')
plt.title("K-Means Clustering with Centroids")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Elbow Method
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
inertias = []
for k in k_values:
    km_model = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    km_model.fit(X)
    inertias.append(km_model.inertia_)

# To Graph
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# Implementing the Calculation of the Silhouette Coefficient
k_values_silhouette = [2, 3, 4, 5, 6, 7, 8, 9, 10]
labels_silhouette = []
silhouette_scores_values = []
for k in k_values_silhouette:
    km_model_silhouette = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    km_model_silhouette.fit(X)
    labels = km_model_silhouette.labels_
    labels_silhouette.append(labels)
    score = silhouette_score(X, labels) 
    silhouette_scores_values.append(score)


# To Graph
plt.plot(k_values_silhouette, silhouette_scores_values, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Coefficient Method')
plt.show()

