import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import ast
import matplotlib.pyplot as plt
from metrics import haversine


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


def scale_gps_features(df, k):
    gps_features = []
    for polyline in df['POLYLINE']:
        polyline = ast.literal_eval(polyline)
        # We only take first and last k points
        if len(polyline) < 2 * k:
            polyline += [polyline[-1]] * (2 * k - len(polyline))
        gps_features.append(polyline[:k] + polyline[-k:])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(gps_features)
    return scaled


def equirectangular_distance(pred, true, R = 6371):
    lambda_pred, phi_pred = pred[:, 0], pred[:, 1]
    lambda_true, phi_true = true[:, 0], true[:, 1]
    x = (lambda_true - lambda_pred) * torch.cos((phi_true + phi_pred) / 2)
    y = phi_true - phi_pred
    return R * torch.sqrt(x ** 2 + y ** 2)


def compute_destination(output, cluster_centroids):
    return torch.matmul(output, cluster_centroids)


def process_test_data(df, k):
    scaled_features_test = scale_gps_features(df, k)
    return torch.tensor(scaled_features_test, dtype=torch.float)


def evaluate_predictions(pred_clusters, true_clusters, cluster_centroids):
    pred_coords = cluster_centroids[pred_clusters]
    true_coords = cluster_centroids[true_clusters]
    distances = haversine(pred_coords, true_coords)
    return distances.mean()


if __name__ == '__main__':
    df = pd.read_csv('../train.csv')
    train_losses = []

    kmeans = KMeans(n_clusters=3392, random_state=42)
    destination_coordinates = ast.literal_eval(df['destination_coordinates'])  # cast to numerical
    df['destination_cluster'] = kmeans.fit_predict(destination_coordinates)
    cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)

    k = 4  # only take into account first and last 4 points of the trajectory
    input_size = 4 * k
    num_clusters = kmeans.n_clusters

    model = TaxiMLP(input_size, num_clusters=num_clusters)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scaled_features = scale_gps_features(df, k)
    inputs = torch.tensor(scaled_features, dtype=torch.float)
    targets = torch.tensor(df['destination_cluster'].values, dtype=torch.long)

    dataset = TensorDataset(inputs, targets)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(5):
        epoch_loss_hav = 0
        for inputs, targets in data_loader:
            outputs = model(inputs)
            destinations = compute_destination(outputs, cluster_centers)
            loss = equirectangular_distance(destinations, targets)  # try with haversine
            loss_haversine = haversine(destinations, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_hav += loss_haversine.item()
            predicted_cluster_indices = outputs.max(1)[1]
            predicted_center_coordinates = cluster_centers[predicted_cluster_indices]
        train_losses.append(epoch_loss_hav / len(data_loader))

    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time (Haversine Distance)')
    plt.legend()
    plt.show()
