import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import ast
import matplotlib.pyplot as plt
from metrics import haversine
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


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


def equirectangular_distance(pred, true, R = 6371):
    lambda_pred, phi_pred = pred[:, 0], pred[:, 1]
    lambda_true, phi_true = true[:, 0], true[:, 1]
    x = (lambda_true - lambda_pred) * torch.cos((phi_true + phi_pred) / 2)
    y = phi_true - phi_pred
    return R * torch.sqrt(x ** 2 + y ** 2)


def compute_destination(output, cluster_centroids):
    return torch.matmul(output, cluster_centroids)


def evaluate_predictions(pred_clusters, true_clusters, cluster_centroids):
    pred_coords = cluster_centroids[pred_clusters]
    true_coords = cluster_centroids[true_clusters]
    distances = haversine(pred_coords, true_coords)
    return distances.mean()


if __name__ == '__main__':
    df2 = pd.read_csv('train_clean.csv')
    df2['END'] = df2['END'].apply(ast.literal_eval)
    kmeans = KMeans(n_clusters=3392, random_state=42)
    destination_coordinates = pd.DataFrame(df2['END'].tolist(), columns=['longitude', 'latitude'])
    kmeans.fit_predict(destination_coordinates)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
    num_clusters = kmeans.n_clusters
    df = pd.read_csv('train_clean_clustered.csv')
    train_losses = []

    targets = torch.tensor(df['CLUSTER'].values, dtype=torch.long)
    df.drop('CLUSTER', axis=1)
    df['POLYLINE'] = df['POLYLINE'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    polylines_tensors = [torch.tensor(pol, dtype=torch.float) for pol in df['POLYLINE'] if pol]

    max_length = max(pol.size(0) for pol in polylines_tensors)
    inputs = pad_sequence(polylines_tensors, batch_first=True)

    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = TaxiMLP(2, num_clusters=num_clusters)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(5):
        epoch_loss_hav = 0
        for inputs, targets in data_loader:
            outputs = model(inputs)
            destinations = compute_destination(outputs, cluster_centers)
            loss = equirectangular_distance(destinations, targets)

            destinations = destinations.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            loss_haversine = haversine(destinations, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_hav += loss_haversine.item()

        print('epoch: ', epoch)
        print('loss: ', epoch_loss_hav)
        train_losses.append(epoch_loss_hav / len(data_loader))

    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time (Haversine Distance)')
    plt.legend()
    plt.show()
