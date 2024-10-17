# methods/window_and_label/window_and_label.py

import pandas as pd
from datetime import timedelta
from methods.helper.helper import datetimeify


def construct_windows(Nw, ti, data, iw, io, dto):
    """Create overlapping data windows for feature extraction.

    Parameters:
    -----------
    Nw : int
        Number of windows to create.
    ti : datetime.datetime
        End of first window.
    data : pandas.DataFrame
        Data to window.
    iw : int
        Number of samples in window.
    io : int
        Number of samples in overlapping section of window.
    dto : datetime.timedelta
        Time delta for non-overlapping section.

    Returns:
    --------
    df : pandas.DataFrame
        Dataframe of windowed data, with 'id' column denoting individual windows.
    window_dates : list
        Datetime objects corresponding to the beginning of each data window.
    """
    dfs = []
    for i in range(Nw):
        start_idx = i * (iw - io)
        end_idx = start_idx + iw
        dfi = data.iloc[start_idx:end_idx].copy()
        if len(dfi) != iw:
            continue
        dfi["id"] = pd.Series([i] * iw, index=dfi.index)
        dfs.append(dfi)
    df = pd.concat(dfs)
    window_dates = [ti + i * dto for i in range(Nw)]
    return df, window_dates
