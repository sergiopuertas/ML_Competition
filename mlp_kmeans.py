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

# Define el número de coordenadas y características adicionales
k = 5
num_coord_features = 4 * k - 2  # 2 * 2 * k coordenadas
num_additional_features = 0  # Reemplaza con el número de características adicionales

# Define tu red neuronal
class TaxiMLP(nn.Module):
    def __init__(self, input_size, hidden_size=500, output_size=2):  # Ajusta output_size según sea necesario
        super(TaxiMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')

        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def equirectangular_distance(pred, true, R=6371):
    # Convertir coordenadas de grados a radianes
    lambda_pred, phi_pred = torch.deg2rad_(pred[:, 0]), torch.deg2rad_(pred[:, 1])
    lambda_true, phi_true = torch.deg2rad_(true[:, 0]), torch.deg2rad_(true[:, 1])

    # Cálculo de la distancia equirectangular
    x = (lambda_true - lambda_pred) * torch.cos((phi_true + phi_pred) / 2)
    y = phi_true - phi_pred
    return R * torch.sqrt(x ** 2 + y ** 2)


# Carga los datos
df = pd.read_csv('processed_data.csv')

y = torch.tensor(df[['END_Lat', 'END_Long']].values, dtype=torch.float)
# Convierte las características a tensores
X = torch.tensor(df.drop(['END_Lat','END_Long','CALL_TYPE','ORIGIN_CALL','ORIGIN_STAND','TAXI_ID','TIMESTAMP','DAY_TYPE'], axis=1).values, dtype=torch.float)
# Crea un conjunto de datos y un cargador de datos
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
# Instancia el modelo y define el optimizador y la función de pérdida
model = TaxiMLP(num_coord_features + num_additional_features)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # Ajusta según sea necesario


model2 = KNeighborsRegressor(15)
optimizer2 = torch.optim.Adam(params=model.parameters(), lr=0.01)

train_losses = []
# Bucle de entrenamiento
for epoch in range(2):  # Ajusta num_epochs según sea necesario
    epoch_loss = 0
    for inputs, targets in data_loader:
        # Forward pass
        outputs = model(inputs)
        if(epoch_loss==0):print(inputs)
        if(epoch_loss==0):print(outputs)
        if(epoch_loss==0):print(targets)
        outputs_numpy = outputs.detach().cpu().numpy()
        targets_numpy = targets.detach().cpu().numpy()

        # Calculate the loss as a numpy array
        loss_numpy = haversine(outputs_numpy, targets_numpy)
        # Convert the numpy loss back to a PyTorch tensor for backward pass
        loss = torch.tensor(loss_numpy, requires_grad=True)
        # Backward y optimización
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        epoch_loss += loss_numpy.mean().item()
    train_losses.append(epoch_loss/len(data_loader))
    print(f'Epoch [{epoch+1}/{5}], Loss: {epoch_loss:.4f}')

# Guarda el modelo entrenado

plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time (Haversine Distance)')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'taxi_mlp_model.pth')

testSet = pd.read_csv('processed_test.csv')

test_trips_ids = list(testSet.index)
y = torch.tensor(testSet[['END_Lat', 'END_Long']].values, dtype=torch.float)
X = torch.tensor(testSet.drop(['END_Lat','END_Long','coord_2','coord_3'], axis=1).values, dtype=torch.float)
outputs = model(X)

# Swap columns (longitude, latitude) -> (latitude, longitude)
destinations = np.stack((outputs[:, 1].detach().numpy(), outputs[:, 0].detach().numpy()), axis=-1)

# Write submission
write_submission(
    trip_ids=test_trips_ids,
    destinations = destinations,
    file_name="example_submission"
)
