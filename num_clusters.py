from sklearn.cluster import KMeans
import pandas as pd
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

df = pd.read_csv('train.csv')
train_losses = []

df['END'] = df['END'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
coordinates = pd.DataFrame(df['END'].tolist(), columns=['longitude', 'latitude'])

range_clusters = range(500, 3500, 1000)
inertias = []

silhouette_scores = []

for n_clusters in range_clusters:
    print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(coordinates)
    inertias.append(kmeans.inertia_)
    if n_clusters > 1:
        score = silhouette_score(coordinates, kmeans.labels_)
        silhouette_scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(range_clusters, inertias, '-o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.xticks(range_clusters)
plt.show()

if silhouette_scores:
    plt.figure(figsize=(8, 4))
    plt.plot(range_clusters[1:], silhouette_scores, '-o')
    plt.title('Silhouette Score For Each k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(range_clusters)
    plt.show()
