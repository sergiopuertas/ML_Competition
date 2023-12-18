import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from metrics import haversine
from example_script import write_submission
import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import torch.nn.functional as F

k = 5
num_coord_features = 4 * k
num_additional_features = 6

# Define tu red neuronal
class TaxiMLP(nn.Module):
    def __init__(self, input_size, hidden_size=500, output_size=2):
        super(TaxiMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def equirectangular_distance(pred, true, R=6371):
    lambda_pred, phi_pred = pred[:, 0], pred[:, 1]
    lambda_true, phi_true = true[:, 0], true[:, 1]
    x = (lambda_true - lambda_pred) * np.cos((phi_true + phi_pred) / 2)
    y = phi_true - phi_pred
    return R * np.sqrt(x**2 + y**2)


if __name__ == '__main__':

    df = pd.read_csv('processed_train.csv')

    y = torch.tensor(df[['END_Lat', 'END_Long']].values, dtype=torch.float)
    X = torch.tensor(df.drop(['END_Lat','END_Long'], axis=1).values, dtype=torch.float)

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = TaxiMLP(num_coord_features + num_additional_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 60
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in data_loader:
            outputs = model(inputs)
            outputs_numpy = outputs.detach().cpu().numpy()
            targets_numpy = targets.detach().cpu().numpy()
            outputs_numpy = np.stack((outputs_numpy[:, 1], outputs_numpy[:, 0]), axis=-1)
            loss_numpy = haversine(outputs_numpy, targets_numpy)
            loss = torch.tensor(loss_numpy, requires_grad=True)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            epoch_loss += loss_numpy.mean().item()
        train_losses.append(epoch_loss/len(data_loader))
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time (Haversine Distance)')
    plt.legend()
    plt.show()

    testSet = pd.read_csv('processed_test.csv')

    test_trips_ids = list(testSet.index)
    y = torch.tensor(testSet[['END_Lat', 'END_Long']].values, dtype=torch.float)
    X = torch.tensor(testSet.drop(['END_Lat','END_Long'], axis=1).values, dtype=torch.float)
    outputs = model(X)

    destinations = np.stack((outputs[:, 1].detach().numpy(), outputs[:, 0].detach().numpy()), axis=-1)
    write_submission(
        trip_ids=test_trips_ids,
        destinations = destinations,
        file_name="sumbission_mlp"
    )
