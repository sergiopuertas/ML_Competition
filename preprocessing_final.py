import pandas as pd
import ast
import sys
k = 5

def process_polyline(polyline):
    coords = ast.literal_eval(polyline) if isinstance(polyline, str) else polyline
    if len(coords) < 2 * k:
        coords += [coords[-1]] * (2 * k - len(coords))
    coords = coords[:2*k]
    flattened_coords = [item for sublist in coords for item in sublist]
    return flattened_coords


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        print("PLease, select a file")
        sys.exit(1)

    data = pd.read_csv(file_name + '.csv')

    data = data[data['MISSING_DATA'] == False]
    data['POLYLINE'] = data['POLYLINE'].apply(ast.literal_eval)
    data['N_POINTS'] = [len(pol) for pol in data.POLYLINE]
    data = data[data.N_POINTS >= 3]
    data.reset_index(drop = True, inplace = True)

    data['START'] = [pol[0] if len(pol) > 0 else [None, None] for pol in data['POLYLINE']]
    data['END'] = [pol[-1] if len(pol) > 0 else [None, None] for pol in data['POLYLINE']]
    data['Processed_POLYLINE'] = data['POLYLINE'].apply(process_polyline)

    num_columns = 4 * k
    polyline_columns = [f'coord_{i}' for i in range(num_columns)]
    data[polyline_columns] = pd.DataFrame(data['Processed_POLYLINE'].tolist(), columns=polyline_columns, index=data.index)

    data['ORIGIN_CALL'].fillna(0, inplace=True)
    data['ORIGIN_STAND'].fillna(0, inplace=True)
    data['CALL_TYPE'] = data['CALL_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2])
    data['DAY_TYPE'] = data['DAY_TYPE'].replace(['A', 'B', 'C'], [0, 1, 2])
    data['TAXI_ID'] -= 20000000
    data['END_Long'] = [eval(polyline)[-1][0] for polyline in data['POLYLINE']]
    data['END_Lat'] = [eval(polyline)[-1][1] for polyline in data['POLYLINE']]
    data["TIMESTAMP"] = pd.to_datetime(data["TIMESTAMP"], unit="s").dt.hour

    data.drop(['POLYLINE', 'Processed_POLYLINE', 'TRIP_ID','END','START','N_POINTS','MISSING_DATA'], axis=1, inplace=True)

    data.to_csv('processed_' + file_name + '.csv', index=False)

