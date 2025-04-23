import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("kmean\Mall_Customers.csv")

# Preview the data
print("🔍 Dataset Head:")
print(df.head())

# Select features for clustering
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualize clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                      c=df['Cluster'], cmap='Set2', s=100, alpha=0.7)
plt.title('Mall Customer Segmentation (k=3)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()

# Print cluster centers (inverse transform to original scale)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("\n📌 Cluster Centers (Approximate):")
for i, center in enumerate(centers):
    print(f"Cluster {i}: Age={center[0]:.1f}, Income={center[1]:.1f}, Spending Score={center[2]:.1f}")

# Analyze clusters
print("\n📊 Cluster Analysis:")
for i in range(3):
    cluster_df = df[df['Cluster'] == i]
    print(f"\nCluster {i} Summary:")
    print(cluster_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe())
