import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from metrics import haversine
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import ast
class TaxiMLP(nn.Module):
    def __init__(self, input_size, hidden_size=500, num_clusters=3392):
        super(TaxiMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_clusters)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def equirectangular_distance(pred, true, R=6371):
    lambda_pred, phi_pred = pred[:, 0], pred[:, 1]
    lambda_true, phi_true = true[:, 0], true[:, 1]
    x = (lambda_true - lambda_pred) * torch.cos((phi_true - phi_pred) / 2)
    y = phi_true - phi_pred
    return R * torch.sqrt(x ** 2 + y ** 2)


def compute_destination(output, cluster_centroids):
    return torch.matmul(output, cluster_centroids)


def evaluate_predictions(pred_clusters, true_clusters, cluster_centroids):
    pred_coords = cluster_centroids[pred_clusters]
    true_coords = cluster_centroids[true_clusters]
    distances = haversine(pred_coords, true_coords)
    return distances.mean()


# We assume some preprocessing has been done to eliminate all unnecessary attributes and
# a new column called CLUSTER (initially empty) has been created. No END_Long or END_Lat necessary
df = pd.read_csv('processed_clustered_train.csv')

kmeans = KMeans(n_clusters=3392, random_state=42)
df['END'] = df['END'].apply(ast.literal_eval)
destination_coordinates = pd.DataFrame(df['END'].tolist(), columns=['longitude', 'latitude'])
df['CLUSTER'] = kmeans.fit_predict(destination_coordinates)
cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
num_clusters = kmeans.n_clusters

y = torch.tensor(df[['CLUSTER']].values, dtype=torch.float)
X = torch.tensor(df.drop(['CLUSTER'], axis=1).values, dtype=torch.float)

dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = TaxiMLP(input_size=2, num_clusters=3392)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

train_losses = []
i = 0

for epoch in range(20):
    epoch_loss_hav = 0
    for inputs, targets in data_loader:
        i+=1
        outputs = model(inputs)
        predicted_cluster_indices = torch.argmax(outputs, dim=1)
        predicted_coords = cluster_centers[predicted_cluster_indices]
        true_coords = cluster_centers[targets]
        loss = equirectangular_distance(predicted_coords, true_coords)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss_hav += loss.item()

    print('epoch: ', epoch)
    print('loss: ', epoch_loss_hav)
    train_losses.append(epoch_loss_hav / len(data_loader))

plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time (Haversine Distance)')
plt.legend()
plt.show()

# Same process for test set and evaluate predictions...
