# methods/feature_extract/feature_extract.py

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from glob import glob
from functools import partial
from multiprocessing import Pool
from methods.helper.helper import datetimeify, makedir
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.transformers import FeatureSelector
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters


def _construct_windows(data, Nw, ti, iw, io, dtw, dto, data_streams, i0=0, i1=None):
    """ Create overlapping data windows for feature extraction.

    Parameters:
    -----------
    data : TremorData
        Object containing tremor data.
    Nw : int
        Number of windows to create.
    ti : datetime.datetime
        End of first window.
    iw : int
        Number of samples in window.
    io : int
        Number of samples in overlapping section of the window.
    dtw : timedelta
        Length of the window.
    dto : timedelta
        Length of the non-overlapping section of the window.
    data_streams : list
        Data streams and transforms from which to extract features.
    i0 : int, optional
        Skip i0 initial windows. Default is 0.
    i1 : int, optional
        Skip i1 final windows. Default is None.

    Returns:
    --------
    df : pandas.DataFrame
        Dataframe of windowed data, with 'id' column denoting individual windows.
    window_dates : list
        Datetime objects corresponding to the beginning of each data window.
    """
    if i1 is None:
        i1 = Nw

    # Get data for windowing period
    start_time = ti - dtw
    end_time = ti + (Nw - 1) * dto
    df = data.get_data(start_time, end_time)[data_streams]

    # Debug: Print data range and length
    print(f"Constructing windows from {start_time} to {end_time}")
    print(f"Data length: {len(df)}")

    # Calculate expected length
    expected_length = (Nw - 1) * (iw - io) + iw
    actual_length = len(df)
    #print(f"Expected data length: {expected_length}, Actual data length: {actual_length}")

    if actual_length < expected_length:
        #print(f"Insufficient data: expected at least {expected_length} samples, got {actual_length}")
        Nw = int((actual_length - iw) / (iw - io)) + 1
        i1 = min(i1, Nw)
        #print(f"Adjusted number of windows to: {Nw}")

    # Create windows
    dfs = []
    window_dates = []
    for i in range(i0, i1):
        start_idx = i * (iw - io)
        end_idx = start_idx + iw
        #print(f"Window {i}: start_idx={start_idx}, end_idx={end_idx}")
        if end_idx > len(df):
            #print(f"Window {i}: end_idx {end_idx} exceeds data length {len(df)}. Skipping.")
            continue
        dfi = df.iloc[start_idx:end_idx]
        actual_length = len(dfi)
        if actual_length != iw:
            print(f"Window {i}: not equal length ({actual_length} != {iw})")
            # パディングを行う場合
            dfi = dfi.reindex(range(start_idx, end_idx))
            dfi['id'] = i
            dfi = dfi.fillna(method='ffill')  # 前方補完
            dfs.append(dfi)
            window_dates.append(ti + i * dto)
            continue
        dfi = dfi.copy()
        dfi['id'] = i
        dfs.append(dfi)
        window_dates.append(ti + i * dto)
    if not dfs:
        raise ValueError("No valid windows found. Check data and window parameters.")
    df_windows = pd.concat(dfs)
    return df_windows, window_dates


def _get_label(data, ts, look_forward):
    """ Compute label vector.

    Parameters:
    -----------
    data : TremorData
        Object containing tremor data.
    ts : array-like
        List of dates to inspect look-forward for eruption.
    look_forward : float
        Length of look-forward in days.

    Returns:
    --------
    ys : list
        Label vector.
    """
    return [data._is_eruption_in(days=look_forward, from_time=t) for t in pd.to_datetime(ts)]


def _extract_features(data, ti, tf, Nw, iw, io, dtw, dto, data_streams, featdir, featfile, look_forward, n_jobs=6, update_feature_matrix=True):
    """ Extract features from windowed data.

    Parameters:
    -----------
    data : TremorData
        Object containing tremor data.
    ti : datetime.datetime
        End of first window.
    tf : datetime.datetime
        End of last window.
    Nw : int
        Number of windows to create.
    iw : int
        Number of samples in window.
    io : int
        Number of samples in overlapping section of the window.
    dtw : timedelta
        Length of the window.
    dto : timedelta
        Length of the non-overlapping section of the window.
    data_streams : list
        Data streams and transforms from which to extract features.
    featdir : str
        Directory to save feature matrices.
    featfile : str
        File path to save feature matrix.
    look_forward : float
        Length of look-forward in days.
    n_jobs : int, optional
        Number of parallel jobs. Default is 6.
    update_feature_matrix : bool, optional
        Flag to update the feature matrix. Default is True.

    Returns:
    --------
    fm : pandas.DataFrame
        tsfresh feature matrix extracted from data windows.
    ys : pandas.DataFrame
        Label vector corresponding to data windows.
    """
    makedir(featdir)

    # Features to compute
    cfp = ComprehensiveFCParameters()

    # Check if feature matrix already exists and what it contains
    if os.path.isfile(featfile):
        existing_fm = pd.read_csv(featfile, index_col=0, parse_dates=['time'], infer_datetime_format=True)
        ti0, tf0 = existing_fm.index[0], existing_fm.index[-1]
        Nw0 = len(existing_fm)
        existing_features = list(set([hd.split('__')[1] for hd in existing_fm.columns]))

        # Determine padding
        pad_left = int((ti0 - ti) / dto) if ti < ti0 else 0
        pad_right = int(((ti + (Nw - 1) * dto) - tf0) / dto) if tf > tf0 else 0
        i0 = abs(pad_left) if pad_left < 0 else 0
        i1 = Nw0 + max([pad_left, 0]) + pad_right

        # Determine new features
        new_features = set(cfp.keys()) - set(existing_features)
        more_cols = bool(new_features)
        if more_cols:
            cfp = {k: v for k, v in cfp.items() if k in new_features}

        # Update feature matrix if needed
        if (more_cols or pad_left > 0 or pad_right > 0) and update_feature_matrix:
            fm = existing_fm.copy()

            # Add new columns
            if more_cols:
                df_new, wd_new = _construct_windows(data, Nw0, ti0, iw, io, dtw, dto, data_streams)
                fm_new = extract_features(df_new, column_id='id', n_jobs=n_jobs, default_fc_parameters=cfp, impute_function=impute)
                fm_new.index = pd.Series(wd_new)
                fm = pd.concat([fm, fm_new], axis=1, sort=False)

            # Add new rows on the left
            if pad_left > 0:
                df_left, wd_left = _construct_windows(data, pad_left, ti, iw, io, dtw, dto, data_streams, i0=0, i1=pad_left)
                fm_left = extract_features(df_left, column_id='id', n_jobs=n_jobs, default_fc_parameters=cfp, impute_function=impute)
                fm_left.index = pd.Series(wd_left)
                fm = pd.concat([fm_left, fm], sort=False)

            # Add new rows on the right
            if pad_right > 0:
                df_right, wd_right = _construct_windows(data, pad_right, ti + (Nw - pad_right) * dto, iw, io, dtw, dto, data_streams, i0=0, i1=pad_right)
                fm_right = extract_features(df_right, column_id='id', n_jobs=n_jobs, default_fc_parameters=cfp, impute_function=impute)
                fm_right.index = pd.Series(wd_right)
                fm = pd.concat([fm, fm_right], sort=False)

            # Save updated feature matrix
            fm.to_csv(featfile, index=True, index_label='time')
            fm = fm.iloc[i0:i1]
        else:
            # Read relevant part of the existing feature matrix
            fm = existing_fm.iloc[i0:i1]
    else:
        # Create feature matrix from scratch
        df_windows, wd = _construct_windows(data, Nw, ti, iw, io, dtw, dto, data_streams)
        fm = extract_features(df_windows, column_id='id', n_jobs=n_jobs, default_fc_parameters=cfp, impute_function=impute)
        fm.index = pd.Series(wd)
        fm.to_csv(featfile, index=True, index_label='time')

    # Compute labels
    ys = pd.DataFrame(_get_label(data, fm.index.values, look_forward=look_forward), columns=['label'], index=fm.index)
    return fm, ys
