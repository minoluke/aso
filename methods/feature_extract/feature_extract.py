# methods/feature_extract/feature_extract.py

import os
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from datetime import timedelta
from methods.helper.helper import datetimeify
from methods.window_and_label.window_and_label import construct_windows


def extract_features_func(data, window_dates, cfp, n_jobs=1):
    """Extract features from windowed data.

    Parameters:
    -----------
    data : pandas.DataFrame
        Windowed data with 'id' column denoting window ids.
    window_dates : list
        List of window dates corresponding to 'id's in data.
    cfp : dict
        Comprehensive feature parameters for tsfresh.
    n_jobs : int
        Number of jobs for parallel processing.

    Returns:
    --------
    fm : pandas.DataFrame
        Feature matrix extracted from data windows.
    """
    fm = extract_features(
        data,
        column_id="id",
        n_jobs=n_jobs,
        default_fc_parameters=cfp,
        impute_function=impute,
    )
    fm.index = pd.Series(window_dates)
    return fm


def construct_features(ti, tf, data, cfp, iw, io, dto, n_jobs=1):
    """Construct features over a period.

    Parameters:
    -----------
    ti : datetime.datetime
        Start time.
    tf : datetime.datetime
        End time.
    data : pandas.DataFrame
        Data to extract features from.
    cfp : dict
        Comprehensive feature parameters for tsfresh.
    iw : int
        Number of samples in window.
    io : int
        Number of samples in overlapping section of window.
    dto : datetime.timedelta
        Time delta for non-overlapping section.
    n_jobs : int
        Number of jobs for parallel processing.

    Returns:
    --------
    fm : pandas.DataFrame
        Feature matrix over the period.
    """
    Nw = int(np.floor(((tf - ti) / dto).total_seconds() / (3600 * 24)))
    df, window_dates = construct_windows(Nw, ti, data, iw, io, dto)
    fm = extract_features_func(df, window_dates, cfp, n_jobs)
    return fm


def get_label(tes, ts, look_forward):
    """Compute label vector.

    Parameters:
    -----------
    tes : list
        List of eruption dates.
    ts : list
        List of dates to inspect look-forward for eruption.
    look_forward : float
        Look-forward period in days.

    Returns:
    --------
    ys : list
        Label vector.
    """
    labels = []
    for t in ts:
        label = 0
        for te in tes:
            if 0 < (te - t).total_seconds() / (3600 * 24) < look_forward:
                label = 1
                break
        labels.append(label)
    return labels
