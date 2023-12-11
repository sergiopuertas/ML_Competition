import numpy as np
import pandas as pd
from ast import literal_eval
from datetime import datetime

from metrics import haversine

import sys

def normalize(df):
    """
    Normalize trajectories from dataframe.
    """
    #print(df.loc[0, 'START'])
    #print(df.loc[0, 'START'].shape)
    #print(np.radians(df.loc[0, 'START']))
    df['DISTANCE'] = [haversine(df.loc[ii, 'START'], df.loc[ii, 'END'])
                      for ii in range(df.shape[0])]
    df.POLYLINE = [df.loc[ii, 'POLYLINE'] - df.loc[ii, 'START']
                   for ii in range(df.shape[0])]
    max_d = max(df.DISTANCE)
    df.POLYLINE = [df.loc[ii, 'POLYLINE'] / max_d
                   for ii in range(df.shape[0])]


data = pd.read_csv(sys.argv[1])

data.POLYLINE = [np.array(literal_eval(pol)) for pol in data.POLYLINE]
data['N_POINTS'] = [len(pol) for pol in data.POLYLINE]
data = data[data.N_POINTS >= 3]
data.reset_index(drop = True, inplace = True)
data['START'] = [pol[0, :].reshape(1, 2) for pol in data.POLYLINE]
data['END'] = [pol[-1, :].reshape(1, 2) for pol in data.POLYLINE]

data['CLUSTER'] = np.empty(data.shape[0])
normalize(data)

data = data.drop('DISTANCE', axis = 1)

data.CALL_TYPE.replace({'A': 1, 'B': 2, 'C': 3})
data.DAY_TYPE.replace({'A': 1, 'B': 2, 'C': 3})

data.POLYLINE = [pol.tolist() for pol in data.POLYLINE]
data.START = [pol.reshape((2,)).tolist() for pol in data.START]
data.END = [pol.reshape((2,)).tolist() for pol in data.END]

data.to_csv(f'{sys.argv[1].split(".")[0]+"_clean.csv"}')

