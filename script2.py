import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from metrics import haversine

# Cargar el dataset
train_data = pd.read_csv('processed_data.csv')
test_data = pd.read_csv('processed_test.csv')
y_test = pd.read_csv('solutions.csv')

# Dividir los datos en características y objetivo
X_train = train_data.drop(['END_Long', 'END_Lat'], axis=1).values
y_train = train_data[['END_Long', 'END_Lat']]

X_test = test_data.drop(['END_Long', 'END_Lat'],axis=1).values
y_test = y_test.drop(['TRIP_ID','PUBLIC'],axis = 1).values

# Escalar los datos (opcional, pero recomendado)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Diccionario de modelos e hiperparámetros
models_hyperparams = {
    'KNN': {
        'model': KNeighborsRegressor,
        'params_list': [{'n_neighbors': k, 'weights': w} for k in [5, 10, 12, 15] for w in ['uniform', 'distance']]
    },
    'DecisionTree': {
        'model': DecisionTreeRegressor,
        'params_list': [{'max_depth': d} for d in [5, 10, 20, 30]]
    },
    'RandomForest': {
        'model': RandomForestRegressor,
        'params_list': [{'n_estimators': n, 'max_depth': d} for n in [10, 50, 100] for d in [10, 15, 20]]
    },
    'SVR': {
        'model': SVR,
        'params_list': [{'C': c, 'kernel': k} for c in [0.1, 1, 10] for k in ['rbf', 'linear']]
    }
}

# Entrenamiento y evaluación de cada modelo
scores = []
for model_name, model_info in models_hyperparams.items():
    for params in model_info['params_list']:
        # Create a model with the given hyperparameters

        model = model_info['model'](**params)
        print(model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.stack((y_pred[:, 1], y_pred[:, 0]), axis=-1)
        loss = haversine(y_pred,y_test)

        scores.append({
            'model': model_name,
            'params': params,
            'loss': loss
        })

# Convertir los resultados en un DataFrame
results_df = pd.DataFrame(scores, columns=['model', 'params', 'loss'])
print(results_df)
results_df.to_csv('results.csv', index=False)

# Graficar los resultados
fig, ax = plt.subplots()
sorted_results = results_df.sort_values(by='loss', ascending=True)

for i, row in sorted_results.iterrows():
    ax.scatter(i, row['mse'], label=f"{row['model']} {row['params']}")

ax.set_xlabel('Index')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Hyperparameter Tuning Results')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()