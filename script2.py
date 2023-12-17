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
models = {
    'KNN': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    },
    'DecisionTree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [5, 10, 15, 20]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [5, 10, 15]
        }
    },
    'SVR': { # support vector machines
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        }
    }
}

# Entrenamiento y evaluación de cada modelo
scores = []
for model_name, mp in models.items():
    print(model_name, mp)
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    loss = haversine(y_train,y_test)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_,
        'loss': loss
    })

# Convertir los resultados en un DataFrame
results_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params', 'loss'])
print(results_df)
results_df.to_csv('results.csv', index=False)

# Graficar los resultados
fig, ax = plt.subplots()
results_df.sort_values(by='loss', ascending=False).plot(x='model', y='loss', kind='barh', ax=ax)
ax.set_xlabel('Loss')
ax.set_title('Model Performance Comparison')
plt.show()
