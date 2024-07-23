# Module 4: Unsupervised Learning

## 1. Clustering
Clustering is an unsupervised learning task where the goal is to group similar instances together based on their characteristics. Scikit-learn provides various clustering algorithms such as KMeans, Hierarchical Clustering, and DBSCAN.

Example using KMeans for clustering:
```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate synthetic data with 3 clusters
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Create an instance of the KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3)

# Fit the model to the data
kmeans.fit(X)

# Get cluster assignments and plot clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red')
plt.show()
```

## 2. Dimensionality Reduction
Dimensionality reduction is another unsupervised learning task where the goal is to reduce the number of features while preserving important information. Scikit-learn provides techniques such as Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.

Example using PCA for dimensionality reduction:
```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Perform PCA for 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the reduced data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.show()
```

These examples demonstrate the usage of supervised and unsupervised learning algorithms provided by scikit-learn. Depending on your specific problem and dataset, you can choose appropriate algorithms and techniques to analyze and model your data effectively.

Absolutely! Let's go through each module one by one:
