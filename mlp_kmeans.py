import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from metrics import haversine
import matplotlib.pyplot as plt


# Define el número de coordenadas y características adicionales
k = 5
num_coord_features = 4 * k  # 2 * 2 * k coordenadas
num_additional_features = 8  # Reemplaza con el número de características adicionales

# Define tu red neuronal
class TaxiMLP(nn.Module):
    def __init__(self, input_size, hidden_size=500, output_size=2):  # Ajusta output_size según sea necesario
        super(TaxiMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

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

# Carga los datos
df = pd.read_csv('processed_data.csv')

y = torch.tensor(df[['coord_18', 'coord_19']].values, dtype=torch.float)
df.drop(['coord_18','coord_19'], axis=1, inplace=True)

# Convierte las características a tensores
X = torch.tensor(df.values, dtype=torch.float)

# Crea un conjunto de datos y un cargador de datos
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Instancia el modelo y define el optimizador y la función de pérdida
model = TaxiMLP(num_coord_features + num_additional_features)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # Ajusta según sea necesario
train_losses = []
# Bucle de entrenamiento
for epoch in range(5):  # Ajusta num_epochs según sea necesario
    epoch_loss = 0
    for inputs, targets in data_loader:
        # Forward pass
        outputs = model(inputs)
        loss = equirectangular_distance(outputs, targets)
        # Backward y optimización
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
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
