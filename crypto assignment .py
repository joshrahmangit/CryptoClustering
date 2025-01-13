import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the crypto data
data_path = '/mnt/data/crypto_market_data.csv'
crypto_data = pd.read_csv(data_path, index_col='coin_id')

# Normalize the data using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(crypto_data)
scaled_crypto_data = pd.DataFrame(scaled_data, 
                                  columns=crypto_data.columns, 
                                  index=crypto_data.index)

# Elbow method to find the best value for k
k_values = list(range(1, 12))
inertia_values = []  # Empty list to store inertia values

# Compute inertia for each value of k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_crypto_data)
    inertia_values.append(kmeans.inertia_)

# Create a dictionary with the data for plotting
elbow_data = {
    "k": k_values,
    "inertia": inertia_values
}

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(elbow_data["k"], elbow_data["inertia"], marker='o', linestyle='-')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Cluster cryptocurrencies with K-Means using the best value for k
best_k = 4  # Optimal k value from the elbow method
kmeans_model = KMeans(n_clusters=best_k, random_state=0)
clusters = kmeans_model.fit_predict(scaled_crypto_data)

# Add the predicted clusters to the original scaled DataFrame
scaled_crypto_data['Cluster'] = clusters

# Create a copy of the original data and add the cluster labels
crypto_data_with_clusters = crypto_data.copy()
crypto_data_with_clusters['Cluster'] = clusters

# Create a scatter plot of the clusters
crypto_data_with_clusters.plot.scatter(
    x='price_change_percentage_24h', 
    y='price_change_percentage_7d', 
    c='Cluster', 
    colormap='viridis', 
    figsize=(10, 6), 
    title='Cryptocurrency Clusters (24h vs 7d Price Changes)'
)
plt.show()

# Optimize clusters with PCA
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_crypto_data.drop(columns=['Cluster']))
pca_df = pd.DataFrame(
    pca_data, 
    columns=['PC1', 'PC2', 'PC3'], 
    index=scaled_crypto_data.index
)

# Display the explained variance ratio
explained_variance = pca.explained_variance_ratio_
total_explained_variance = explained_variance.sum()
print(f"Explained Variance Ratio: {explained_variance}")
print(f"Total Explained Variance: {total_explained_variance}")

# Create a new DataFrame with PCA data and retain the coin_id index
pca_crypto_data = pca_df.copy()
print(pca_crypto_data.head())

# Elbow method to find the best value for k using PCA data
pca_k_values = list(range(1, 12))
pca_inertia_values = []  # Empty list to store inertia values for PCA data

for k in pca_k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pca_crypto_data)
    pca_inertia_values.append(kmeans.inertia_)

# Create a dictionary with the data for PCA elbow plotting
pca_elbow_data = {
    "k": pca_k_values,
    "inertia": pca_inertia_values
}

# Plot the elbow curve for PCA data
plt.figure(figsize=(8, 5))
plt.plot(pca_elbow_data["k"], pca_elbow_data["inertia"], marker='o', linestyle='-')
plt.title("Elbow Method for Optimal k (PCA Data)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(pca_k_values)
plt.grid(True)
plt.show()

# Cluster cryptocurrencies with K-Means using the PCA data
pca_best_k = 4  # Optimal k value from the PCA elbow method
pca_kmeans_model = KMeans(n_clusters=pca_best_k, random_state=0)
pca_clusters = pca_kmeans_model.fit_predict(pca_crypto_data)

# Add the predicted clusters to the PCA DataFrame
pca_crypto_data['Cluster'] = pca_clusters

# Create a copy of the PCA DataFrame and add the cluster labels
pca_crypto_data_with_clusters = pca_crypto_data.copy()

# Create a scatter plot of the PCA clusters
pca_crypto_data_with_clusters.plot.scatter(
    x='PC1', 
    y='PC2', 
    c='Cluster', 
    colormap='viridis', 
    figsize=(10, 6), 
    title='PCA Cryptocurrency Clusters'
)
plt.show()

# Determine the weights of each feature on each principal component
pca_components = pd.DataFrame(
    pca.components_, 
    columns=scaled_crypto_data.drop(columns=['Cluster']).columns, 
    index=['PC1', 'PC2', 'PC3']
).T

# Find the strongest positive or negative influences for each principal component
strongest_influences = pca_components.abs().idxmax()
strongest_weights = pca_components.lookup(row_labels=strongest_influences.index, col_labels=strongest_influences)

print("Strongest Influences per Principal Component:")
print(strongest_influences)
print("\nCorresponding Weights:")
print(strongest_weights)

# Add summary of the strongest influences
summary = """
Strongest Influences per Principal Component:
- PC1: {PC1} ({PC1_weight:.4f})
- PC2: {PC2} ({PC2_weight:.4f})
- PC3: {PC3} ({PC3_weight:.4f})
""".format(
    PC1=strongest_influences['PC1'], PC1_weight=strongest_weights[0],
    PC2=strongest_influences['PC2'], PC2_weight=strongest_weights[1],
    PC3=strongest_influences['PC3'], PC3_weight=strongest_weights[2]
)
print(summary)
