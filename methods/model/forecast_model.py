# methods/model/forecast_model.py

import os
import pandas as pd
import numpy as np
import gc
import shutil
from glob import glob
from datetime import datetime, timedelta
from functools import partial

from methods.data.data import TremorData
from methods.window_and_label.window_and_label import construct_windows
from methods.feature_extract.feature_extract import (
    extract_features_func,
    construct_features,
    get_label,
)
from methods.train.train import (
    train,
    exclude_dates,
    collect_features,
    load_data,
)
from methods.test.test import forecast
from methods.helper.helper import get_classifier, datetimeify
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters


class ForecastModel(object):
    """Object for training and running forecast models.

    Attributes:
    -----------
    data : TremorData
        Object containing tremor data.
    dtw : datetime.timedelta
        Length of window.
    dtf : datetime.timedelta
        Length of look-forward.
    dto : datetime.timedelta
        Length of non-overlapping section of window.
    iw : int
        Number of samples in window.
    io : int
        Number of samples in overlapping section of window.
    ti_model : datetime.datetime
        Beginning of model analysis period.
    tf_model : datetime.datetime
        End of model analysis period.
    exclude_dates : list
        List of time windows to exclude during training.
    n_jobs : int
        Number of CPUs to use for parallel tasks.
    rootdir : str
        Repository location on file system.
    plotdir : str
        Directory to save forecast plots.
    modeldir : str
        Directory to save forecast models.
    featdir : str
        Directory to save feature matrices.
    featfile : str
        File to save feature matrix to.
    preddir : str
        Directory to save forecast model predictions.
    consensusdir : str
        Directory to save consensus forecasts.

    Methods:
    --------
    train
        Construct classifier models.
    forecast
        Use classifier models to forecast eruption likelihood.
    """

    def __init__(
        self,
        window,
        overlap,
        look_forward,
        data_streams,
        ti=None,
        tf=None,
        root=None,
        od=None,
        n_jobs=1,
    ):
        self.window = window
        self.overlap = overlap
        self.look_forward = look_forward
        self.data_streams = data_streams
        self.data = TremorData()
        self.n_jobs = n_jobs

        # Validate data streams
        if any([d not in self.data.df.columns for d in self.data_streams]):
            raise ValueError(
                "Data streams restricted to any of {}".format(self.data.df.columns)
            )

        # Set analysis period
        if ti is None:
            ti = self.data.ti
        if tf is None:
            tf = self.data.tf
        self.ti_model = datetimeify(ti)
        self.tf_model = datetimeify(tf)
        if self.tf_model > self.data.tf:
            raise ValueError(
                "Model end date '{:s}' beyond data range '{:s}'".format(
                    str(self.tf_model), str(self.data.tf)
                )
            )
        if self.ti_model < self.data.ti:
            raise ValueError(
                "Model start date '{:s}' predates data range '{:s}'".format(
                    str(self.ti_model), str(self.data.ti)
                )
            )

        self.dtw = timedelta(days=self.window)
        self.dtf = timedelta(days=self.look_forward)
        self.dt = timedelta(days=1.0)
        self.dto = (1.0 - self.overlap) * self.dtw
        self.iw = int(self.window)
        self.io = int(self.overlap * self.iw)
        if self.io == self.iw:
            self.io -= 1

        self.window = self.iw * 1.0
        self.dtw = timedelta(days=self.window)
        if self.ti_model - self.dtw < self.data.ti:
            self.ti_model = self.data.ti + self.dtw
        self.overlap = self.io * 1.0 / self.iw
        self.dto = (1.0 - self.overlap) * self.dtw

        self.exclude_dates = []
        self.update_feature_matrix = True

        # Define root directory first
        self.rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        # Naming convention and file system attributes
        if root is None:
            self.root = "fm_{:3.2f}wndw_{:3.2f}ovlp_{:3.2f}lkfd".format(
                self.window, self.overlap, self.look_forward
            )
            self.root += "_" + ("{:s}-" * len(self.data_streams))[:-1].format(
                *sorted(self.data_streams)
            )
        else:
            self.root = root

        # Use 'od' for organizing save directories
        if od is not None:
            self.od = od
            self.consensusdir = os.path.join(self.rootdir, "save", "consensus", self.od)
        else:
            self.od = "default"
            self.consensusdir = os.path.join(self.rootdir, "save", "consensus")

        self.plotdir = os.path.join(self.rootdir, "save", "figures", self.root)
        self.modeldir = os.path.join(self.rootdir, "save", "rawdata", "model", self.root)
        self.featdir = os.path.join(self.rootdir, "save", "rawdata", "feature")
        self.featfile = os.path.join(self.featdir, "{}_features.csv".format(self.root))
        self.preddir = os.path.join(self.rootdir, "save", "rawdata", "test", self.root)
        self.consensusdir = os.path.join(
            self.consensusdir, f"{self.window}_{self.look_forward}"
        )

        # Create necessary directories
        os.makedirs(self.plotdir, exist_ok=True)
        os.makedirs(self.modeldir, exist_ok=True)
        os.makedirs(self.featdir, exist_ok=True)
        os.makedirs(self.preddir, exist_ok=True)
        os.makedirs(self.consensusdir, exist_ok=True)

    def train(
        self,
        cv=0,
        ti=None,
        tf=None,
        Nfts=20,
        Ncl=100,
        retrain=False,
        classifier="DT",
        random_seed=0,
        exclude_dates_list=[],
    ):
        """Construct classifier models.

        Parameters:
        -----------
        cv : int
            Cross-validation fold index.
        ti : str, datetime.datetime
            Beginning of training period.
        tf : str, datetime.datetime
            End of training period.
        Nfts : int
            Number of most-significant features to use in classifier.
        Ncl : int
            Number of classifier models to train.
        retrain : boolean
            Use saved models (False) or train new ones.
        classifier : str
            String denoting which classifier to train.
        random_seed : int
            Random seed for reproducibility.
        exclude_dates : list
            List of time windows to exclude during training.
        """
        self.classifier = classifier

        # Initialize training interval
        self.ti_train = self.ti_model if ti is None else datetimeify(ti)
        self.tf_train = self.tf_model if tf is None else datetimeify(tf)
        if self.ti_train - self.dtw < self.data.ti:
            self.ti_train = self.data.ti + self.dtw

        # Load feature matrix and label vector
        fM, ys = self._load_features_and_labels()

        # Exclude specified dates
        fM, ys = exclude_dates(fM, ys, exclude_dates_list)
        if ys.shape[0] != fM.shape[0]:
            raise ValueError("Dimensions of feature matrix and label vector do not match")

        # Select training subset
        inds = (ys.index >= self.ti_train) & (ys.index < self.tf_train)
        fM = fM.loc[inds]
        ys = ys.loc[inds]

        # Train models
        train(
            fM,
            ys,
            Nfts,
            self.modeldir,
            self.classifier,
            Ncl=Ncl,
            retrain=retrain,
            random_seed=random_seed,
            n_jobs=self.n_jobs,
            exclude_dates_list=exclude_dates_list,
        )

        # Collect features
        all_fts_path = os.path.join(self.modeldir, "all.fts")
        collect_features(self.modeldir, save=all_fts_path)

        # Organize feature files into consensus directory
        ob_folder = os.path.join(self.consensusdir, self.od)
        wl_lfl_folder = os.path.join(ob_folder, f"{self.window}_{self.look_forward}")
        os.makedirs(wl_lfl_folder, exist_ok=True)

        new_all_fts_path = os.path.join(wl_lfl_folder, f"{cv}_all.fts")

        # Copy all.fts to the new directory
        if os.path.exists(all_fts_path):
            shutil.copy(all_fts_path, new_all_fts_path)

    def forecast(
        self,
        cv=0,
        ti=None,
        tf=None,
        recalculate=False,
        use_model=None,
    ):
        """Use classifier models to forecast eruption likelihood.

        Parameters:
        -----------
        cv : int
            Cross-validation fold index.
        ti : str, datetime.datetime
            Beginning of forecast period.
        tf : str, datetime.datetime
            End of forecast period.
        recalculate : bool
            Flag indicating forecast should be recalculated.
        use_model : None or str
            Optionally pass path to pre-trained model directory in 'models'.

        Returns:
        --------
        consensus : pd.DataFrame
            The model consensus, indexed by window date.
        """
        self._use_model = use_model

        self.ti_forecast = self.ti_model if ti is None else datetimeify(ti)
        self.tf_forecast = self.tf_model if tf is None else datetimeify(tf)
        if self.tf_forecast > self.data.tf:
            self.tf_forecast = self.data.tf
        if self.ti_forecast - self.dtw < self.data.ti:
            self.ti_forecast = self.data.ti + self.dtw

        # Load feature matrix for forecast period
        fM, _ = self._load_features_and_labels()

        # Forecast
        forecast_df = forecast(
            fM,
            self.modeldir,
            self.preddir,
            n_jobs=self.n_jobs,
            recalculate=recalculate,
            use_model=self._use_model,
        )

        # Save consensus forecast
        consensus_path = os.path.join(self.consensusdir, f"{cv}_consensus.csv")
        forecast_df.to_csv(consensus_path, index=True, index_label="time")

        return forecast_df

    def _load_features_and_labels(self):
        """Load or compute features and labels."""
        os.makedirs(self.featdir, exist_ok=True)
        # Check if feature file exists
        if os.path.isfile(self.featfile):
            # Load existing features
            fM = pd.read_csv(
                self.featfile,
                index_col=0,
                parse_dates=["time"],
                infer_datetime_format=True,
            )
        else:
            # Construct features
            cfp = ComprehensiveFCParameters()  # Customize if needed
            data = self.data.get_data(self.ti_model - self.dtw, self.tf_model)
            fM = construct_features(
                self.ti_model,
                self.tf_model,
                data[self.data_streams],
                cfp,
                self.iw,
                self.io,
                self.dto,
                n_jobs=self.n_jobs,
            )
            fM.to_csv(self.featfile, index=True, index_label="time")

        # Get labels
        ys = pd.Series(
            get_label(self.data.tes, fM.index, self.look_forward),
            index=fM.index,
        )
        ys.name = "label" 
        return fM, ys
