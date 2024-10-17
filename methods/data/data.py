# methods/data/data.py

import os
import pandas as pd
from datetime import datetime
from inspect import getfile, currentframe

from methods.helper.helper import datetimeify


class TremorData(object):
    """Object to manage acquisition and processing of seismic data.

    Attributes:
    -----------
    df : pandas.DataFrame
        Time series of tremor data and transforms.
    t0 : datetime.datetime
        Beginning of data range.
    t1 : datetime.datetime
        End of data range.

    Methods:
    --------
    get_data
        Return tremor data in requested date range.
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(getfile(currentframe())))
        self.file = os.path.join(base_dir, "data", "tremor_data.dat")
        self._assess()

    def __repr__(self):
        if self.exists:
            tm = [self.ti.year, self.ti.month, self.ti.day]
            tm += [self.tf.year, self.tf.month, self.tf.day]
            return "TremorData:{:d}/{:02d}/{:02d} to {:d}/{:02d}/{:02d}".format(*tm)
        else:
            return "no data"

    def _assess(self):
        """Load existing file and check date range of data."""
        base_dir = os.path.dirname(os.path.dirname(getfile(currentframe())))
        eruptive_file_path = os.path.join(base_dir, "data", "eruptive_periods.txt")

        # ファイルの存在確認
        if os.path.exists(eruptive_file_path):
            try:
                # Get eruptions
                with open(eruptive_file_path, "r") as fp:
                    # ファイルから行ごとに読み込み、改行を取り除き、日付を変換
                    self.tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
                print("Eruption data loaded successfully.")
            except Exception as e:
                print(f"Error while reading the file: {e}")
        else:
            print(f"File not found: {eruptive_file_path}")
            
        # Check if data file exists
        self.exists = os.path.isfile(self.file)
        if not self.exists:
            t0 = datetime(2010, 1, 1)
            t1 = datetime(2010, 1, 2)
            self.update(t0, t1)
        # Check date of latest data in file
        self.df = pd.read_csv(
            self.file, index_col=0, parse_dates=[0], infer_datetime_format=True
        )
        self.ti = self.df.index[0]
        self.tf = self.df.index[-1]

    def get_data(self, ti=None, tf=None):
        """Return tremor data in requested date range.

        Parameters:
        -----------
        ti : str, datetime.datetime
            Date of first data point (default is earliest data).
        tf : str, datetime.datetime
            Date of final data point (default is latest data).

        Returns:
        --------
        df : pandas.DataFrame
            Data object truncated to requested date range.
        """
        # Set date range defaults
        if ti is None:
            ti = self.ti
        if tf is None:
            tf = self.tf

        # Convert datetime format
        ti = datetimeify(ti)
        tf = datetimeify(tf)

        # Subset data
        inds = (self.df.index >= ti) & (self.df.index < tf)
        return self.df.loc[inds]

    def _is_eruption_in(self, days, from_time):
        """Binary classification of eruption imminence.

        Parameters:
        -----------
        days : float
            Length of look-forward.
        from_time : datetime.datetime
            Beginning of look-forward period.

        Returns:
        --------
        label : int
            1 if eruption occurs in look-forward, 0 otherwise
        """
        for te in self.tes:
            if 0 < (te - from_time).total_seconds() / (3600 * 24) < days:
                return 1.0
        return 0.0
