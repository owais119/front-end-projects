import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load Wholesale Customers Data from UCI
df = pd.read_csv("heirachial_clutering\Wholesale customers data.csv")

# Display first few rows of the dataset
print(df.head())

# Feature scaling (Standardize the data)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)



#agglometerive clustering 

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Choose number of clusters, say 3 clusters
agg_clust = AgglomerativeClustering(n_clusters=3, linkage='complete')
agg_clust_labels = agg_clust.fit_predict(scaled_data)

# Add cluster labels to the original dataframe
df['Agglomerative_Cluster'] = agg_clust_labels


# Dendrogram generation to choose optimal number of clusters
plt.figure(figsize=(10, 7))
sch.dendrogram(sch.linkage(scaled_data, method='complete'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()


# visualize 

from sklearn.decomposition import PCA

# Reduce data to 2D using PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=agg_clust_labels, cmap='viridis', s=50)
plt.title('Agglomerative Clustering (Complete Linkage)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


#compare


from sklearn.cluster import KMeans

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

# Add KMeans labels to dataframe
df['KMeans_Cluster'] = kmeans_labels

# Visualize KMeans clusters using PCA
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title('K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Compare clustering results
print("Agglomerative Clustering labels:")
print(df['Agglomerative_Cluster'].value_counts())
print("\nK-Means Clustering labels:")
print(df['KMeans_Cluster'].value_counts())
