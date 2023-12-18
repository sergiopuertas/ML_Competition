import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from example_script import write_submission
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler
from metrics import haversine

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
        'params_list': [{'n_estimators': n, 'max_depth': d} for n in [10, 50] for d in [10, 15, 20]]
    },
    'Ridge': {
        'model': Ridge,
        'params_list': [{'alpha': a} for a in [0.1, 1, 10, 100, 1000]]
    },
}

if __name__ == '__main__':
    train_data = pd.read_csv('processed_train.csv')
    test_data = pd.read_csv('processed_test.csv')
    y_test = pd.read_csv('solutions.csv')

    X_train = train_data.drop(['END_Long', 'END_Lat'], axis=1).values
    y_train = train_data[['END_Long', 'END_Lat']]

    X_test = test_data.drop(['END_Long', 'END_Lat'], axis=1).values
    y_test = y_test.drop(['TRIP_ID', 'PUBLIC'], axis=1).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scores = []
    for model_name, model_info in models_hyperparams.items():
        for params in model_info['params_list']:
            model = model_info['model'](**params)
            print(model)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = np.stack((y_pred[:, 1], y_pred[:, 0]), axis=-1)
            loss = haversine(y_pred, y_test)
            scores.append({
                'model': model_name,
                'params': params,
                'loss': loss
            })

        gc.collect()

    results_df = pd.DataFrame(scores, columns=['model', 'params', 'loss'])
    print(results_df)
    results_df.to_csv('results.csv', index=False)

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

    best_model_info = results_df.sort_values(by='loss').iloc[0]
    best_model_name = best_model_info['model']
    best_params = best_model_info['params']

    best_model = models_hyperparams[best_model_name]['model'](**best_params)
    best_model.fit(X_train, y_train)
    best_pred = best_model.predict(X_test)

    write_submission(
        trip_ids=len(X_test),
        destinations=best_pred,
        file_name="best_model_submission.csv"
    )
