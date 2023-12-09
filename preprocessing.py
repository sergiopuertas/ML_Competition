import numpy as np
from ast import literal_eval
from datetime import datetime

from metrics import haversine

import sys
import logging

### Change format of polyline
def polyline_manipulation(df):
    """
    Changes attribute POLYLINE from a string representation of a list, into a
    proper list of numerical values.

    Adds columns with the initial and final coordinates, as well as amount of
    points in the GPS series.
    """
    df['N_POINTS'] = 0
    df['START'] = [np.zeros(2) for ii in range(df.shape[0])]
    df['END'] = [np.zeros(2) for ii in range(df.shape[0])]
    df['T_TIME'] = 0.
    df.POLYLINE = [np.array(literal_eval(pol)) for pol in df.POLYLINE]
    for ii in range(df.shape[0]):
        try:
            pol = df.iloc[ii].POLYLINE
            if len(pol) != 0:
                df.loc[ii, 'N_POINTS'] = len(pol)
                df.loc[ii, 'START'][:] = np.array(pol[0])
                df.loc[ii, 'END'][:] = np.array(pol[-1])
                df.loc[ii, 'T_TIME'] = len(pol) * 15
            else:
                out = ('POLYLINE: Trip %d has no data in the POLYLINE, '
                       'and %s MISSING_DATA flag.')
                logging.debug(out
                              % (df.iloc[ii].TRIP_ID, df.iloc[ii].MISSING_DATA))
                df.loc[ii, 'N_POINTS'] = 0
                df.loc[ii, 'START'] = np.nan
                df.loc[ii, 'END'] = np.nan
                df.loc[ii, 'T_TIME'] = np.nan
        except Exception as e:
            logging.debug('POLYLINE: Unexpected problem in iteration '
                          '%d: \'%s\'\n' % (ii, e))
            raise e
    logging.info('Size before removing null trips: %d' % df.shape[0])
    df = df[df.N_POINTS > 0]
    logging.info('Size after removing null trips: %d\n' % df.shape[0])
    return df


### Calculate tau
def haversine(p1, p2):
    """
    Local function to calculate Haversine distance that can take individual
    points instead of arrays of points.

    Might be deleted later depending on how the code develops.
    """
    lat1 = np.radians(p1[0])
    lon1 = np.radians(p1[1])
    lat2 = np.radians(p2[0])
    lon2 = np.radians(p2[1])

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

    d = 2 * 6371 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return d


def tau(df):
    df['TAU'] = 0.
    for ii in range(df.shape[0]):
        try:
            # Calculate distance between each consecute step of the POLYLINE
            logging.debug('TAU: Calculation for %s' % df.loc[ii, 'TRIP_ID'])
            step_distance = [haversine(df.loc[ii, 'POLYLINE'][jj-1],
                                       df.loc[ii, 'POLYLINE'][jj])
                             for jj in range(1, len(df.loc[ii, 'POLYLINE']))]
            overall_distance = haversine(df.loc[ii, 'START'],
                                         df.loc[ii, 'END'])
            df.loc[ii, 'TAU'] = sum(step_distance, overall_distance)
        except Exception as e:
            logging.debug('TAU: calculation problem with trip %s\n'
                          % df.loc[ii, 'TRIP_ID'])
            df.loc[ii, 'TAU'] = np.nan
            raise e

    return df


### Change hour format to readable
def time_manipulation(df):
    """
    Original timestamps are in Unix Timestamp format. This means very little
    in terms of analysis, and is not readable. This function adds a new column
    with the time in human-readable format.
    """
    def _time(df):
        for ii in range(df.shape[0]):
            try:
                yield datetime.utcfromtimestamp(df.loc[ii, 'TIMESTAMP'])
            except Exception as e:
                logging.debug('TIMESTAMP: \'%s\' in element %d: %d'
                              % (e,
                                 df.loc[ii].TRIP_ID,
                                 df.loc[ii, 'TIMESTAMP'])
                              )
                yield np.nan
                raise e

    df['TIME'] = list(_time(df))
    return df


### Find outliers from distances to discard them
# Returns a bool if it needs to be discarded
def remove_outliers(df, c_name):
    """
    Removes outliers of dataframe from a column with identifier c_name.
    """
    logging.info('\nRemoving outliers for column %s' % c_name)

    o_size = df.shape[0]
    logging.info('Datapoints before outlier screening: %d' % o_size)

    q25 = df[c_name].quantile(0.25)
    q75 = df[c_name].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5*iqr
    upper_bound = q75 + 1.5*iqr
    logging.debug('OUTLIERS: Lower bound for outliers: %f' % lower_bound)
    logging.debug('OUTLIERS: Upper bound for outliers: %f' % upper_bound)
    
    df = df[df[c_name] > lower_bound]
    df = df[df[c_name] < upper_bound]
    
    e_size = df.shape[0]
    logging.info('Datapoints after outlier screening: %d' % e_size)
    logging.info('Removal of outliers from column %s: %f %% removed'
                 % (c_name, ((1 - e_size/o_size) * 100)))
    return df


### Find candidates for GPS failure
def identify_gps_gaps(timeseries, threshold, update_time):
    """
    Get GPS data. For each contiguous pair of points, caluclate the distance
    and speed required to traverse it considering 15s between datapoints.
    If speed exceeds threshold, flag indices for interpolation.

    Input:
    - timeseries: Iterable with GPS signals from a single trip.
    - threshold: speed to determine excess speed due to gaps.
    - update_time: amount of seconds between consecutive gps signals.

    Output:
    - 0 if no gaps were detected.
    - flagged: iterable of indices between which problems were identified.
    """
    flagged = []
    started = False
    pair = [None, None]
    for ii, coords in enumerate(timeseries):
        # Checking for boundary case at the end of the list
        if ii == len(timeseries) - 1:
            if not started:
                break
            else:
                pair[1] = ii
                flagged.append(pair.copy())
                break

        other = timeseries[ii + 1]
        dist = haversine(coords, other) # in km
        delta_t = update_time / 3600
        speed = dist / delta_t
        # If speed limit is not and has not been exceded, nothing happens
        if speed <= threshold and not started:
            pass
        # If speed limit has not stopped being exceded, nothing happens
        elif speed > threshold and started:
            pass
        # Signal the start of speed limit exceeding
        elif speed > threshold and not started:
            started = True
            pair[0] = ii
        # Signal the end of speed limit exceeding, and add to list 
        elif speed <= threshold and started:
            started = False
            pair[1] = ii
            flagged.append(pair.copy())

    if len(flagged) == 0:
        return 0
    else:
        return flagged


def gaps(df, update_time = 15):
    """
    Checks for GPS gaps for all trajectories with a progressively lower speed
    threshold. Stops when all rows with MISSING_DATA == True are properly
    labeled.
    """
    thresholds = np.arange(150, 50, -10)
    for speed in thresholds:
        df['GAPS'] = [identify_gps_gaps(pl, speed, update_time)
                      for pl in df.POLYLINE]
        labels = df[df.MISSING_DATA == True].GAPS
        if np.all(labels != 0):
            logging.info('GAPS: All missing data trips properly labeled '
                         'with speed  %d' % speed)
            return df
        else:
            logging.info('\nGAPS: MISSING_DATA trips with True values '
                         'mislabeled at speed %d' % speed)


def interpolate_trajectory(df):
    """
    Checks the row of the Dataframe. If there are no identified gaps, returns
    the normal POLYLINE. If there are, does linear interpolation between the
    gaps and returns the new "corrected" POLYLINE.
    """
    def _interpolate(pol, indices):
        # Catch cases where the whole trajectory requires interpolation
        if (len(indices) == 1 and indices[0][0] == 0 and
            indices[0][1] == len(pol) - 1):
            logging.info('INTERPOLATION: Case with whole trajectory needs '
                         'to be interpolated. Returning \'None\' to drop '
                         'row.')
            return None

        ori_copy = pol.copy()
        interpolated = pol[:indices[0][0]]
        for gap in indices:
            start, end = gap
            interpolated = np.concatenate((interpolated,
                                           foo(ori_copy, start, end)
                                          )
                                         )
        return interpolated

    def foo(original_trajectory, start, end, update_time = 15):
        if start == 0:
            ps = np.nan
            ns = (haversine(original_trajectory[end],
                            original_trajectory[end + 1]) / update_time)
        elif end == len(original_trajectory) - 1:
            ps = (haversine(original_trajectory[start],
                            original_trajectory[start - 1]) / update_time)
            ns = np.nan
        else:
            ps = (haversine(original_trajectory[start],
                            original_trajectory[start - 1]) / update_time)
            ns = (haversine(original_trajectory[end],
                            original_trajectory[end + 1]) / update_time)
        avg_speed = np.nanmean([ps, ns])
        dist = haversine(original_trajectory[start], original_trajectory[end])
        npoints = round(((dist / avg_speed) / (3600)) / update_time)

        x_intp = np.interp(np.linspace(0, npoints), [0, npoints],
                           original_trajectory[[start, end]][:, 0])
        y_intp = np.interp(np.linspace(0, npoints), [0, npoints],
                           original_trajectory[[start, end]][:, 1])
        x_intp = x_intp.reshape((len(x_intp), 1))
        y_intp = y_intp.reshape((len(y_intp), 1))
        return np.concatenate((x_intp, y_intp), axis = 1)

    df['POLYLINE_CLEAN'] = np.empty(df.shape[0], dtype = 'object')
    for ii in range(df.shape[0]):
        if df.loc[ii, 'GAPS'] == 0:
            df.loc[ii, 'POLYLINE_CLEAN'] = [df.loc[ii, 'POLYLINE']]
        else:
            try:
                out = ('INTERPOLATION: initiated at trip %s')
                logging.debug(out % df.loc[ii, 'TRIP_ID'])
                ip = _interpolate(df.loc[ii, 'POLYLINE'], df.loc[ii, 'GAPS'])
                if ip is not None:
                    df.loc[ii, 'POLYLINE_CLEAN'] = [ip]
            except Exception as e:
                logging.error('INTERPOLATION: failed interpolation at trip '
                              '%s' % df.loc[ii, 'TRIP_ID'])
                raise e

    logging.info('NONE CLEANUP: %d elements before cleanup' % df.shape[0])
    df.dropna(subset = 'POLYLINE_CLEAN', inplace = True)
    logging.info('NONE CLEANUP: %d elements after cleanup' % df.shape[0])


def cluster(df):
    df['CLUSTER'] = None
    return


if __name__ == '__main__':
    import pandas as pd
    database = sys.argv[1]
    print(database)
    logging.basicConfig(filename = f'{database.split('.')[0]}.log',
                        encoding = 'utf-8', filemode = 'w',
                        level = logging.DEBUG
                       )
    outname = database.split('.')[0] + '_clean.pckl'
    print(outname)

    data = pd.read_csv(database)

    # Change polyline format and add additional columns depending on it
    # Remove empty trips
    try:
        data = polyline_manipulation(data)
    except Exception as e:
        data.to_pickle(outname)
        logging.error('Failed at polyline manipulation \'%s\'\n'
                      'Check log for details' % e)
        raise e
        exit()

    try:
        logging.info('SUCCESS: Passed POLYLINE manipulation.')
        gaps(data, 15)
    except Exception as e:
        data.to_pickle(outname)
        logging.error('Failed at gap identification \'%s\'\n'
                      'Check log for details.' % e)
        raise e
        exit()

    try:
        logging.info('SUCCESS: Passed GAPS indentification.')
        interpolate_trajectory(data)
    except Exception as e:
        data.to_pickle(outname)
        logging.error('Failed at POLYLINE interpolation \'%s\'\n'
                      'Check log for details.' % e)
        raise e
        exit()

    # Update N_POINTS and T_TIME with POLYLINE_CLEAN
    logging.info('SUCCESS: Completed interpolation of POLYLINEs')
    data.N_POINTS = [len(pl[0]) for pl in data.POLYLINE_CLEAN]
    data = data[data.N_POINTS > 3]
    data = remove_outliers(data, 'N_POINTS')
    data.T_TIME = [npoints * 15 for npoints in data.N_POINTS]

    # Reset indexes
    data.reset_index(drop = True, inplace = True)

    # Calculate TAU values
    try:
        logging.info('SUCCESS: Completed outlier removal')
        data = tau(data)
    except Exception as e:
        data.to_pickle(outname)
        logging.error('Could not do TAU stuff.')
        raise e
        exit()

    # Change TIME format
    try:
        logging.info('SUCCESS: Passed TAU calculation')
        data = time_manipulation(data)
    except Exception as e:
        data.to_pickle(outname)
        logging.error('Could not change TIME format.')
        raise e
        exit()

    logging.info('SUCCESS: Finished TIME formatting')
    data.CALL_TYPE.replace({'A': 1, 'B': 2, 'C': 3})
    data.DAY_TYPE.replace({'A': 1, 'B': 2, 'C': 3})
    data.to_pickle(outname)

