# --- IMPORTS ---
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "6"  

# --- LOAD DATA ---
iris_dataset = load_iris()
X_iris = iris_dataset.data
y_iris_true = iris_dataset.target

print("Shape of X_iris:", X_iris.shape)
print("Shape of y_iris_true:", y_iris_true.shape)
print("Feature names:", iris_dataset.feature_names)
print("Class names (species):", iris_dataset.target_names)
print("First 5 rows of X_iris:\n", X_iris[:5])
print("First 5 labels of y_iris_true:", y_iris_true[:5])

# Elbow Method
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
inertias = []

for k in k_values:
    km_model = KMeans(n_clusters=k, init='k-means++', n_init='auto' , random_state=42)
    km_model.fit(X_iris)
    inertias.append(km_model.inertia_)

# To Graph
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Iris Data')
plt.show()

# Implementing the Calculation of the Silhouette Coefficient
k_values_silhouette = [2, 3, 4, 5, 6, 7, 8, 9, 10]
silhouette_scores_values = []
for k in k_values_silhouette:
    km_model_silhouette = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    km_model_silhouette.fit(X_iris)
    score = silhouette_score(X_iris, km_model_silhouette.labels_)
    silhouette_scores_values.append(score)

# To Graph
plt.plot(k_values_silhouette, silhouette_scores_values, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Coefficient Method for Iris Data')
plt.show()


kmeans_model = KMeans(n_clusters=3, init='k-means++', n_init='auto', random_state=42)
kmeans_model.fit(X_iris)

y_kmeans_iris_pred = kmeans_model.labels_
centroides_iris_kmeans = kmeans_model.cluster_centers_
inertia_iris_kmeans = kmeans_model.inertia_
print(inertia_iris_kmeans)

plt.figure(figsize=(14, 6))  # Adjust figure size to fit two plots side by side

# Plot 1: Clusters found by K-Means
plt.subplot(1, 2, 1)  # (1 row, 2 columns, first plot)
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_kmeans_iris_pred, cmap='viridis', s=50)  # Colored by K-Means predictions
plt.scatter(centroides_iris_kmeans[:, 0], centroides_iris_kmeans[:, 1], s=200, c='red', marker='X', label='K-Means Centroids')  # Emphasize centroids
plt.title("K-Means Clusters (K=3) on Iris (Sepals)")
plt.xlabel(iris_dataset.feature_names[0])
plt.ylabel(iris_dataset.feature_names[1])
plt.legend()

# Plot 2: True Iris Species
plt.subplot(1, 2, 2)  # (1 row, 2 columns, second plot)
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris_true, cmap='viridis', s=50)  # Colored by true labels
plt.title("True Iris Species (Sepals)")
plt.xlabel(iris_dataset.feature_names[0])
plt.ylabel(iris_dataset.feature_names[1])
# You could add a legend here mapping y_iris_true values to species names, if needed

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
