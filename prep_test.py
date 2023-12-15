import pandas as pd
import ast
from metrics import haversine
# Define value of k
k = 5

# Load DataFrame
df = pd.read_csv('test.csv')


def process_polyline(polyline):
    coords = ast.literal_eval(polyline) if isinstance(polyline, str) else polyline
    if len(coords) < 2 * k:
        coords += [coords[-1]] * (2 * k - len(coords))
    coords = coords[:2*k]  # + coords[-k:]
    flattened_coords = [item for sublist in coords for item in sublist]  # Aplanar la lista
    return flattened_coords

'''
def normalize(df):
    df['DISTANCE'] = [haversine(df.loc[ii, 'START'], df.loc[ii, 'END'])
                      for ii in range(df.shape[0])]
    df.POLYLINE = [df.loc[ii, 'POLYLINE'] - df.loc[ii, 'START']
                   for ii in range(df.shape[0])]
    max_d = max(df.DISTANCE)
    df.POLYLINE = [df.loc[ii, 'POLYLINE'] / max_d
                   for ii in range(df.shape[0])]
'''

# Aplica la función a cada fila en la columna POLYLINE
df['Processed_POLYLINE'] = df['POLYLINE'].apply(process_polyline)
#normalize(df)

# Convierte la lista de coordenadas en columnas individuales
num_columns = 4 * k  # 2 coordenadas por punto, 2k puntos en total
polyline_columns = [f'coord_{i}' for i in range(num_columns)]
df[polyline_columns] = pd.DataFrame(df['Processed_POLYLINE'].tolist(), columns=polyline_columns, index=df.index)

df['END_Long'] = [eval(polyline)[-1][0] for polyline in df['POLYLINE']]
df['END_Lat'] = [eval(polyline)[-1][1] for polyline in df['POLYLINE']]

'''
# Procesa las demás columnas
df['ORIGIN_CALL'].fillna(0, inplace=True)
df['ORIGIN_STAND'].fillna(0, inplace=True)
df['CALL_TYPE'] = df['CALL_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2],inplace=True)
df['DAY_TYPE'] = df['DAY_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2],inplace=True)
df['TAXI_ID'] -= 20000000

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], unit="s").dt.hour
# Elimina las columnas que ya no necesitas
df.drop(['POLYLINE', 'Processed_POLYLINE', 'TRIP_ID','coord_0','coord_1','CLUSTER','END','START','N_POINTS','Unnamed: 0','MISSING_DATA'], axis=1, inplace=True)
'''
df.drop(['POLYLINE', 'Processed_POLYLINE', 'TRIP_ID','CALL_TYPE','ORIGIN_CALL','ORIGIN_STAND','TAXI_ID','TIMESTAMP','DAY_TYPE','MISSING_DATA'], axis=1, inplace=True)

# Guarda el DataFrame procesado
df.to_csv('processed_test.csv', index=False)
