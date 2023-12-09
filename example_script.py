import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

def write_submission(trip_ids, destinations, file_name="submission"):
    """
    This function writes a submission csv file given the trip ids, 
    and the predicted destinations .

    Parameters
    ----------
    trip_id : List of Strings
        List of trip ids (e.g., "T1").
    destinations : NumPy Array of Shape (n_samples, 2) with float values
        Array of destinations (latitude and longitude) for each trip.
    file_name : String
        Name of the submission file to be saved.
        Default: "submission".
    """
    n_samples = len(trip_ids)
    assert destinations.shape == (n_samples, 2)

    submission = pd.DataFrame(
        data={
            'LATITUDE': destinations[:, 0],
            'LONGITUDE': destinations[:, 1],
        },
        columns=["LATITUDE", "LONGITUDE"],
        index=trip_ids,
    )

    # Write file
    submission.to_csv(file_name + ".csv", index_label="TRIP_ID")


def load_data(csv_path):
    """
    Reads a CSV file (train or test) and returns the data contained.

    Parameters
    ----------
    csv_path : String
        Path to the CSV file to be read.
        e.g., "train.csv"

    Returns
    -------
    data : Pandas DataFrame 
        Data read from CSV file.
    n_samples : Integer
        Number of rows (samples) in the dataset.
    """
    data = pd.read_csv(csv_path, index_col="TRIP_ID")

    return data, len(data)


def dummy_preprocessing(data):
    data = data[data['POLYLINE'] != '[]']
    data = data[data['MISSING_DATA'] == False][['CALL_TYPE', 'DAY_TYPE', 'POLYLINE']][:10000]
    data['CALL_TYPE'] = data['CALL_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2])
    data['DAY_TYPE'] = data['DAY_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2])
    data['START_Long'] = [eval(polyline)[0][0] for polyline in data['POLYLINE']]
    data['START_Lat'] = [eval(polyline)[0][1] for polyline in data['POLYLINE']]
    data['END_Long'] = [eval(polyline)[-1][0] for polyline in data['POLYLINE']]
    data['END_Lat'] = [eval(polyline)[-1][1] for polyline in data['POLYLINE']]
    data = data.drop('POLYLINE', axis=1)

    X, y = data.drop(['END_Long', 'END_Lat'], axis=1), data[['END_Long', 'END_Lat']]

    return X, y


# This example script will produce the file "example_submission.csv".
if __name__ == "__main__":
    # Train set
    train_data, n_trip_train = load_data("../train.csv")
    print(f"Train data shape: {train_data.shape}")

    # Train set
    test_data, n_trip_test = load_data("../test.csv")
    print(f"Test data shape: {test_data.shape}")
    test_trips_ids = list(test_data.index)

    # If you want to convert the timestamp to datetime format,
    # you can use this code:
    # data["TIMESTAMP"] = pd.to_datetime(data["TIMESTAMP"], unit="s")

    # Training
    # This is a dummy example solution.
    # It uses a decision tree regressor on a small subset of the training data.
    # It only uses the day type, call type and start position to predict the destination.
    X_train, y_train = dummy_preprocessing(train_data)

    clf = DecisionTreeRegressor()
    clf.fit(X_train, y_train)

    X_test, _ = dummy_preprocessing(test_data)
    destinations = clf.predict(X_test)
    
    # Swap columns (longitude, latitude) -> (latitude, longitude)
    destinations = np.stack((destinations[:, 1], destinations[:, 0]), axis=-1)

    # Write submission
    write_submission(
        trip_ids=test_trips_ids, 
        destinations=destinations,
        file_name="example_submission"
    )
