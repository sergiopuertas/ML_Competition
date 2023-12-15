import pandas as pd
import ast

# Define el valor de k
k = 5

# Carga el DataFrame
df = pd.read_csv('train_clean.csv')

def process_polyline(polyline):
    coords = ast.literal_eval(polyline) if isinstance(polyline, str) else polyline
    if len(coords) < 2 * k:
        coords += [coords[-1]] * (2 * k - len(coords))
    coords = coords[:k] + coords[-k:]
    flattened_coords = [item for sublist in coords for item in sublist]  # Aplanar la lista
    return flattened_coords

# Aplica la función a cada fila en la columna POLYLINE
df['Processed_POLYLINE'] = df['POLYLINE'].apply(process_polyline)

# Convierte la lista de coordenadas en columnas individuales
num_columns = 4 * k  # 2 coordenadas por punto, 2k puntos en total
polyline_columns = [f'coord_{i}' for i in range(num_columns)]
df[polyline_columns] = pd.DataFrame(df['Processed_POLYLINE'].tolist(), columns=polyline_columns, index=df.index)

# Procesa las demás columnas
df['ORIGIN_CALL'].fillna(0, inplace=True)
df['ORIGIN_STAND'].fillna(0, inplace=True)
df['CALL_TYPE'] = df['CALL_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2],inplace=True)
df['DAY_TYPE'] = df['DAY_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2],inplace=True)
df['MISSING_DATA'] = df['MISSING_DATA'].replace(['True', 'False'], [1, 0],inplace=True)

# Elimina las columnas que ya no necesitas
df.drop(['POLYLINE', 'Processed_POLYLINE', 'TRIP_ID','coord_0','coord_1','CLUSTER','START','END','N_POINTS','Unnamed: 0'], axis=1, inplace=True)


# Guarda el DataFrame procesado
df.to_csv('processed_data.csv', index=False)
