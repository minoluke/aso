# general imports
import os, sys, shutil, warnings, gc, joblib
import numpy as np
from datetime import datetime, timedelta, date
from copy import copy
from matplotlib import pyplot as plt
from inspect import getfile, currentframe
from glob import glob
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from multiprocessing import Pool
from textwrap import wrap
from time import time
from scipy.integrate import cumtrapz
from scipy.signal import stft
from scipy.optimize import curve_fit
from corner import corner
from functools import partial
from fnmatch import fnmatch
import shutil

# feature recognition imports
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.transformers import FeatureSelector
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from imblearn.under_sampling import RandomUnderSampler

# classifier imports
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

all_classifiers = ['DT']
_MONTH = timedelta(days=365.25/12)
_DAY = timedelta(days=1.)

makedir = lambda name: os.makedirs(name, exist_ok=True)

class TremorData(object):
    """ Object to manage acquisition and processing of seismic data.
        
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
        self.file = os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data','tremor_data.dat'])
        self._assess()
    def __repr__(self):
        if self.exists:
            tm = [self.ti.year, self.ti.month, self.ti.day]
            tm += [self.tf.year, self.tf.month, self.tf.day]
            return 'TremorData:{:d}/{:02d}/{:02d} to {:d}/{:02d}/{:02d}'.format(*tm)
        else:
            return 'no data'

    def _assess(self):
        """ Load existing file and check date range of data.
        """
        # get eruptions
        with open(os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['data','eruptive_periods.txt']),'r') as fp:
            self.tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        # check if data file exists
        self.exists = os.path.isfile(self.file)
        if not self.exists:
            t0 = datetime(2010,1,1)
            t1 = datetime(2010,1,2)
            self.update(t0,t1)
        # check date of latest data in file
        self.df = pd.read_csv(self.file, index_col=0, parse_dates=[0,], infer_datetime_format=True)
        self.ti = self.df.index[0]
        self.tf = self.df.index[-1]

    def _is_eruption_in(self, days, from_time):
        """ Binary classification of eruption imminence.

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
            if 0 < (te-from_time).total_seconds()/(3600*24) < days:
                return 1.
        return 0.

    def get_data(self, ti=None, tf=None):
        """ Return tremor data in requested date range.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Date of first data point (default is earliest data).
            tf : str, datetime.datetime
                Date of final data point (default is latest data).

            Returns:
            --------
            df : pandas.DataFrame
                Data object truncated to requsted date range.
        """
        # set date range defaults
        if ti is None:
            ti = self.ti
        if tf is None:
            tf = self.tf

        # convert datetime format
        ti = datetimeify(ti)
        tf = datetimeify(tf)

        # subset data
        inds = (self.df.index>=ti)&(self.df.index<tf)
        return self.df.loc[inds]
  
class ForecastModel(object):
    """ Object for train and running forecast models.
        
        Constructor arguments:
        ----------------------
        window : float
            Length of data window in days.
        overlap : float
            Fraction of overlap between adjacent windows. Set this to 1. for overlap of entire window minus 1 data point.
        look_forward : float
            Length of look-forward in days.
        ti : str, datetime.datetime
            Beginning of analysis period. If not given, will default to beginning of tremor data.
        tf : str, datetime.datetime
            End of analysis period. If not given, will default to end of tremor data.
        data_streams : list
            Data streams and transforms from which to extract features. Options are 'X', 'diff_X', 'log_X', 'inv_X', and 'stft_X' 
            where X is one of 'rsam', 'mf', 'hf', or 'dsar'.            
        root : str 
            Naming convention for forecast files. If not given, will default to 'fm_*Tw*wndw_*eta*ovlp_*Tlf*lkfd_*ds*' where
            Tw is the window length, eta is overlap fraction, Tlf is look-forward and ds are data streams.

        Attributes:
        -----------
        data : TremorData
            Object containing tremor data.
        dtw : datetime.timedelta
            Length of window.
        dtf : datetime.timedelta
            Length of look-forward.
        dt : datetime.timedelta
            Length between data samples (10 minutes).
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
        ti_train : datetime.datetime
            Beginning of model training period.
        tf_train : datetime.datetime
            End of model training period.
        ti_forecast : datetime.datetime
            Beginning of model forecast period.
        tf_forecast : datetime.datetime
            End of model forecast period.
        drop_features : list
            List of tsfresh feature names or feature calculators to drop during training.
            Facilitates manual dropping of correlated features.
        exclude_dates : list
            List of time windows to exclude during training. Facilitates dropping of eruption 
            windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
            ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.
        use_only_features : list
            List of tsfresh feature names or calculators that training will be restricted to.
        compute_only_features : list
            List of tsfresh feature names or calcluators that feature extraction will be 
            restricted to.
        update_feature_matrix : bool
            Set this True in rare instances you want to extract feature matrix without the code
            trying first to update it.
        n_jobs : int
            Number of CPUs to use for parallel tasks.
        rootdir : str
            Repository location on file system.
        plotdir : str
            Directory to save forecast plots.
        modeldir : str
            Directory to save forecast models (pickled sklearn objects).
        featdir : str
            Directory to save feature matrices.
        featfile : str
            File to save feature matrix to.
        preddir : str
            Directory to save forecast model predictions.

        Methods:
        --------
        _detect_model
            Checks whether and what models have already been run.
        _construct_windows
            Create overlapping data windows for feature extraction.
        _extract_features
            Extract features from windowed data.
        _get_label
            Compute label vector.
        _load_data
            Load feature matrix and label vector.
        _drop_features
            Drop columns from feature matrix.
        _exclude_dates
            Drop rows from feature matrix and label vector.
        _collect_features
            Aggregate features used to train classifiers by frequency.
        _model_alerts
            Compute issued alerts for model consensus.
        get_features
            Return feature matrix and label vector for a given period.
        train
            Construct classifier models.
        forecast
            Use classifier models to forecast eruption likelihood.
        hires_forecast
            Construct forecast at resolution of data.
        plot_forecast
            Plot model forecast.
        plot_accuracy
            Plot performance metrics for model.
        plot_features
            Plot frequency of extracted features by most significant.
        plot_feature_correlation
            Corner plot of feature correlation.
    """
    def __init__(self, window, overlap, look_forward,data_streams, ti=None, tf=None, root=None, od=None):
        self.window = window
        self.overlap = overlap
        self.look_forward = look_forward
        self.data_streams = data_streams
        self.data = TremorData()
        #if any(['_' in ds for ds in data_streams]):self.data._compute_transforms()
        if any([d not in self.data.df.columns for d in self.data_streams]):
            raise ValueError("data restricted to any of {}".format(self.data.df.columns))
        if ti is None: ti = self.data.ti
        if tf is None: tf = self.data.tf
        self.ti_model = datetimeify(ti)
        self.tf_model = datetimeify(tf)
        if self.tf_model > self.data.tf:
            raise ValueError("Model end date '{:s}' beyond data range '{:s}'".format(self.tf_model, self.data.tf))
        if self.ti_model < self.data.ti:
            raise ValueError("Model start date '{:s}' predates data range '{:s}'".format(self.ti_model, self.data.ti))
        self.dtw = timedelta(days=self.window)
        self.dtf = timedelta(days=self.look_forward)
        self.dt = timedelta(days=1.)
        self.dto = (1.-self.overlap)*self.dtw
        self.iw = int(self.window)   
        self.od = od        
        self.io = int(self.overlap*self.iw)      
        if self.io == self.iw: self.io -= 1

        self.window = self.iw*1.
        self.dtw = timedelta(days=self.window)
        if self.ti_model - self.dtw < self.data.ti:
            self.ti_model = self.data.ti+self.dtw
        self.overlap = self.io*1./self.iw
        self.dto = (1.-self.overlap)*self.dtw
        
        self.drop_features = []
        self.exclude_dates = []
        self.use_only_features = []
        self.compute_only_features = []
        self.update_feature_matrix = True
        self.n_jobs = 6

        # naming convention and file system attributes
        if root is None:
            self.root = 'fm_{:3.2f}wndw_{:3.2f}ovlp_{:3.2f}lkfd'.format(self.window, self.overlap, self.look_forward)
            self.root += '_'+((('{:s}-')*len(self.data_streams))[:-1]).format(*sorted(self.data_streams))
        else:
            self.root = root
        self.rootdir = os.sep.join(getfile(currentframe()).split(os.sep)[:-2])
        self.plotdir = r'{:s}/plots/{:s}'.format(self.rootdir, self.root)
        self.modeldir = r'{:s}/models/{:s}'.format(self.rootdir, self.root)
        self.featdir = r'{:s}/features'.format(self.rootdir, self.root)
        self.featfile = r'{:s}/{:s}_features.csv'.format(self.featdir, self.root)
        self.preddir = r'{:s}/predictions/{:s}'.format(self.rootdir, self.root)
        self.consensusdir = r'{:s}/consensus/'.format(self.rootdir)
    # private helper methods
    def _detect_model(self):
        """ Checks whether and what models have already been run.
        """
        fls = glob(self._use_model+os.sep+'*.fts')
        if len(fls) == 0:
            raise ValueError("no feature files in '{:s}'".format(self._use_model))

        inds = [int(float(fl.split(os.sep)[-1].split('.')[0])) for fl in fls if ('all.fts' not in fl)]
        if max(inds) != (len(inds) - 1):
            raise ValueError("feature file numbering in '{:s}' appears not consecutive".format(self._use_model))
        
        self.classifier = []
        for classifier in all_classifiers:
            model = get_classifier(classifier)[0]
            pref = type(model).__name__
            if all([os.path.isfile(self._use_model+os.sep+'{:s}_{:04d}.pkl'.format(pref,ind)) for ind in inds]):
                self.classifier = classifier
                return
        raise ValueError("did not recognise models in '{:s}'".format(self._use_model))
    def _construct_windows(self, Nw, ti, i0=0, i1=None):
        """ Create overlapping data windows for feature extraction.

            Parameters:
            -----------
            Nw : int
                Number of windows to create.
            ti : datetime.datetime
                End of first window.
            i0 : int
                Skip i0 initial windows.
            i1 : int
                Skip i1 final windows.

            Returns:
            --------
            df : pandas.DataFrame
                Dataframe of windowed data, with 'id' column denoting individual windows.
            window_dates : list
                Datetime objects corresponding to the beginning of each data window.
        """
        if i1 is None:
            i1 = Nw

        # get data for windowing period
        df = self.data.get_data(ti-self.dtw, ti+(Nw-1)*self.dto)[self.data_streams]
        #print(df)

        # create windows
        dfs = []
        for i in range(i0, i1):
            dfi = df[:].iloc[i*(self.iw-self.io):i*(self.iw-self.io)+self.iw]
            if len(dfi) != self.iw:
                print("not equal")
            try:
                dfi['id'] = pd.Series(np.ones(self.iw, dtype=int)*i, index=dfi.index)
            except ValueError:
                print('hi')
            dfs.append(dfi)
        df = pd.concat(dfs)
        window_dates = [ti + i*self.dto for i in range(Nw)]
        return df, window_dates[i0:i1]
    def _extract_features(self, ti, tf):
        """ Extract features from windowed data.

            Parameters:
            -----------
            ti : datetime.datetime
                End of first window.
            tf : datetime.datetime
                End of last window.

            Returns:
            --------
            fm : pandas.Dataframe
                tsfresh feature matrix extracted from data windows.
            ys : pandas.Dataframe
                Label vector corresponding to data windows

            Notes:
            ------
            Saves feature matrix to $rootdir/features/$root_features.csv to avoid recalculation.
        """
        makedir(self.featdir)

        # number of windows in feature request
        Nw = int(np.floor(((tf-ti)/self.dt)/(self.iw-self.io)))

        # features to compute
        cfp = ComprehensiveFCParameters()
        if self.compute_only_features:
            cfp = dict([(k, cfp[k]) for k in cfp.keys() if k in self.compute_only_features])
        else:
            # drop features if relevant
            _ = [cfp.pop(df) for df in self.drop_features if df in list(cfp.keys())]

        # check if feature matrix already exists and what it contains
        if os.path.isfile(self.featfile):
            t = pd.to_datetime(pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], usecols=['time'], infer_datetime_format=True).index.values)
            ti0,tf0 = t[0],t[-1]
            Nw0 = len(t)
            hds = pd.read_csv(self.featfile, index_col=0, nrows=1)
            hds = list(set([hd.split('__')[1] for hd in hds]))

            # option 1, expand rows
            pad_left = int((ti0-ti)/self.dto)# if ti < ti0 else 0
            pad_right = int(((ti+(Nw-1)*self.dto)-tf0)/self.dto)# if tf > tf0 else 0
            i0 = abs(pad_left) if pad_left<0 else 0
            i1 = Nw0 + max([pad_left,0]) + pad_right
            
            # option 2, expand columns
            existing_cols = set(hds)        # these features already calculated, in file
            new_cols = set(cfp.keys()) - existing_cols     # these features to be added
            more_cols = bool(new_cols)
            all_cols = existing_cols|new_cols
            cfp = ComprehensiveFCParameters()
            cfp = dict([(k, cfp[k]) for k in cfp.keys() if k in all_cols])

            # option 3, expand both
            if any([more_cols, pad_left > 0, pad_right > 0]) and self.update_feature_matrix:
                fm = pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], infer_datetime_format=True)
                if more_cols:
                    # expand columns now
                    df0, wd = self._construct_windows(Nw0, ti0)
                    cfp0 = ComprehensiveFCParameters()
                    cfp0 = dict([(k, cfp0[k]) for k in cfp0.keys() if k in new_cols])
                    fm2 = extract_features(df0, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp0, impute_function=impute)
                    fm2.index = pd.Series(wd)
                    
                    fm = pd.concat([fm,fm2], axis=1, sort=False)

                # check if updates required because training period expanded
                    # expanded earlier
                if pad_left > 0:
                    df, wd = self._construct_windows(Nw, ti, i1=pad_left)
                    fm2 = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp, impute_function=impute)
                    fm2.index = pd.Series(wd)
                    fm = pd.concat([fm2,fm], sort=False)
                    # expanded later
                if pad_right > 0:
                    df, wd = self._construct_windows(Nw, ti, i0=Nw - pad_right)
                    fm2 = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp, impute_function=impute)
                    fm2.index = pd.Series(wd)
                    fm = pd.concat([fm,fm2], sort=False)
                
                # write updated file output
                fm.to_csv(self.featfile, index=True, index_label='time')
                # trim output
                fm = fm.iloc[i0:i1]    
            else:
                # read relevant part of matrix
                fm = pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], infer_datetime_format=True, header=0, skiprows=range(1,i0+1), nrows=i1-i0)
        else:
            # create feature matrix from scratch   
            df, wd = self._construct_windows(Nw, ti)
            fm = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp, impute_function=impute)
            fm.index = pd.Series(wd)
            fm.to_csv(self.featfile, index=True, index_label='time')
            
        ys = pd.DataFrame(self._get_label(fm.index.values), columns=['label'], index=fm.index)
        return fm, ys
    def _get_label(self, ts):
        """ Compute label vector.

            Parameters:
            -----------
            t : datetime like
                List of dates to inspect look-forward for eruption.

            Returns:
            --------
            ys : list
                Label vector.
        """
        return [self.data._is_eruption_in(days=self.look_forward, from_time=t) for t in pd.to_datetime(ts)]
    def _load_data(self, ti, tf):
        """ Load feature matrix and label vector.

            Parameters:
            -----------
            ti : str, datetime
                Beginning of period to load features.
            tf : str, datetime
                End of period to load features.

            Returns:
            --------
            fM : pd.DataFrame
                Feature matrix.
            ys : pd.DataFrame
                Label vector.
        """
        # return pre loaded
        try:
            if ti == self.ti_prev and tf == self.tf_prev:
                return self.fM, self.ys
        except AttributeError:
            pass

        # read from CSV file
        try:
            t = pd.to_datetime(pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], usecols=['time'], infer_datetime_format=True).index.values)
            if (t[0] <= ti) and (t[-1] >= tf):
                self.ti_prev = ti
                self.tf_prev = tf
                fM,ys = self._extract_features(ti,tf)
                self.fM = fM
                self.ys = ys
                return fM,ys
        except FileNotFoundError:
            pass

        # range checking
        if tf > self.data.tf:
            raise ValueError("Model end date '{:s}' beyond data range '{:s}'".format(tf, self.data.tf))
        if ti < self.data.ti:
            raise ValueError("Model start date '{:s}' predates data range '{:s}'".format(ti, self.data.ti))
        
        # divide training period into years
        ts = [datetime(*[yr, 1, 1, 0, 0, 0]) for yr in list(range(ti.year+1, tf.year+1))]
        if ti - self.dtw < self.data.ti:
            ti = self.data.ti + self.dtw
        ts.insert(0,ti)
        ts.append(tf)

        for t0,t1 in zip(ts[:-1], ts[1:]):
            print('feature extraction {:s} to {:s}'.format(t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d')))
            fM,ys = self._extract_features(ti,t1)

        self.ti_prev = ti
        self.tf_prev = tf
        self.fM = fM
        self.ys = ys
        return fM, ys

    def _exclude_dates(self, X, y, exclude_dates):
        """ Drop rows from feature matrix and label vector.

            Parameters:
            -----------
            X : pd.DataFrame
                Matrix to drop columns.
            y : pd.DataFrame
                Label vector.
            exclude_dates : list
                List of time windows to exclude during training. Facilitates dropping of eruption 
                windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.

            Returns:
            --------
            Xr : pd.DataFrame
                Reduced matrix.
            yr : pd.DataFrame
                Reduced label vector.
        """
        self.exclude_dates = exclude_dates
        if len(self.exclude_dates) != 0:
            for exclude_date_range in self.exclude_dates:
                t0,t1 = [datetimeify(dt) for dt in exclude_date_range]
                inds = (y.index<t0)|(y.index>=t1)
                X = X.loc[inds]
                y = y.loc[inds]
        return X,y
    def _collect_features(self, save=None):
        """ Aggregate features used to train classifiers by frequency.

            Parameters:
            -----------
            save : None or str
                If given, name of file to save feature frequencies. Defaults to all.fts
                if model directory.

            Returns:
            --------
            labels : list
                Feature names.
            freqs : list
                Frequency of feature appearance in classifier models.
        """
        makedir(self.modeldir)
        if save is None:
            save = '{:s}/all.fts'.format(self.modeldir)
        
        feats = []
        fls = glob('{:s}/*.fts'.format(self.modeldir))
        for i,fl in enumerate(fls):
            if fl.split(os.sep)[-1].split('.')[0] in ['all','ranked']: continue
            with open(fl) as fp:
                lns = fp.readlines()
            feats += [' '.join(ln.rstrip().split()[1:]) for ln in lns]               

        labels = list(set(feats))
        freqs = [feats.count(label) for label in labels]
        labels = [label for _,label in sorted(zip(freqs,labels))][::-1]
        freqs = sorted(freqs)[::-1]
        # write out feature frequencies
        with open(save, 'w') as fp:
            _ = [fp.write('{:d},{:s}\n'.format(freq,ft)) for freq,ft in zip(freqs,labels)]
        return labels, freqs
        
    # public methods
    def get_features(self, ti=None, tf=None, n_jobs=1, drop_features=[], compute_only_features=[]):
        """ Return feature matrix and label vector for a given period.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of period to extract features (default is beginning of model analysis).
            tf : str, datetime.datetime
                End of period to extract features (default is end of model analysis).
            n_jobs : int
                Number of cores to use.
            compute_only_features : list
                tsfresh feature names of calculators to return in matrix.
            
            Returns:
            --------
            fM : pd.DataFrame
                Feature matrix.
            ys : pd.Dataframe
                Label vector.
        """
        # initialise training interval
        self.drop_features = drop_features
        self.compute_only_features = compute_only_features
        self.n_jobs = n_jobs
        ti = self.ti_model if ti is None else datetimeify(ti)
        tf = self.tf_model if tf is None else datetimeify(tf)
        return self._load_data(ti, tf)
    def train(self, cv=0, ti=None, tf=None, Nfts=20, Ncl=100, retrain=False, classifier="DT", random_seed=0,
            drop_features=[], n_jobs=6, exclude_dates=[], use_only_features=[]):
        """ Construct classifier models.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of training period (default is beginning model analysis period).
            tf : str, datetime.datetime
                End of training period (default is end of model analysis period).
            Nfts : int
                Number of most-significant features to use in classifier.
            Ncl : int
                Number of classifier models to train.
            retrain : boolean
                Use saved models (False) or train new ones.
            classifier : str, list
                String or list of strings denoting which classifiers to train (see options below.)
            random_seed : int
                Set the seed for the undersampler, for repeatability.
            n_jobs : int
                CPUs to use when training classifiers in parallel.
            exclude_dates : list
                List of time windows to exclude during training. Facilitates dropping of eruption 
                windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.

            Classifier options:
            -------------------
            DT - Decision Tree
        """
        self.classifier = classifier
        self.exclude_dates = exclude_dates
        self.use_only_features = use_only_features
        self.n_jobs = n_jobs
        makedir(self.modeldir)

        # initialise training interval
        self.ti_train = self.ti_model if ti is None else datetimeify(ti)
        self.tf_train = self.tf_model if tf is None else datetimeify(tf)
        if self.ti_train - self.dtw < self.data.ti:
            self.ti_train = self.data.ti+self.dtw
        
        # check if any model training required
        if not retrain:
            run_models = False
            pref = type(get_classifier(self.classifier)[0]).__name__ 
            for i in range(Ncl):         
                if not os.path.isfile('{:s}/{:s}_{:04d}.pkl'.format(self.modeldir, pref, i)):
                    run_models = True
            if not run_models:
                return # not training required
        else:
            # delete old model files
            _ = [os.remove(fl) for fl in  glob('{:s}/*'.format(self.modeldir))]

        # get feature matrix and label vector
        fM, ys = self._load_data(self.ti_train, self.tf_train)

        # manually drop features (columns)
        fM = self._drop_features(fM, drop_features)

        # manually select features (columns)
        if len(self.use_only_features) != 0:
            use_only_features = [df for df in self.use_only_features if df in fM.columns]
            fM = fM[use_only_features]
            Nfts = len(use_only_features)+1

        # manually drop windows (rows)
        fM, ys = self._exclude_dates(fM, ys, exclude_dates)
        if ys.shape[0] != fM.shape[0]:
            raise ValueError("dimensions of feature matrix and label vector do not match")
        
        # select training subset
        inds = (ys.index>=self.ti_train)&(ys.index<self.tf_train)
        fM = fM.loc[inds]
        ys = ys['label'].loc[inds]

        # set up model training
        if self.n_jobs > 1:
            p = Pool(self.n_jobs)
            mapper = p.imap
        else:
            mapper = map
        f = partial(train_one_model, fM, ys, Nfts, self.modeldir, self.classifier, retrain, random_seed)

        # train models with glorious progress bar
        for i, _ in enumerate(mapper(f, range(Ncl))):
            cf = (i+1)/Ncl
            print(f'building models: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
        if self.n_jobs > 1:
            p.close()
            p.join()
        
        # free memory
        del fM
        gc.collect()
        self._collect_features()

        all_fts_path = os.path.join(self.modeldir, 'all.fts')
    
        ob_folder = os.path.join(self.consensusdir, self.od)
        wl_lfl_folder = os.path.join(ob_folder, f"{self.window}_{self.look_forward}")
        makedir(wl_lfl_folder)
        
        new_all_fts_path = os.path.join(wl_lfl_folder, f"{cv}_all.fts")
        
        # ファイルを新しいディレクトリにコピー
        if os.path.exists(all_fts_path):
            shutil.copy(all_fts_path, new_all_fts_path)
        
    def forecast(self,cv=0, ti=None, tf=None, recalculate=False, use_model=None, n_jobs=6):
        """ Use classifier models to forecast eruption likelihood.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of forecast period (default is beginning of model analysis period).
            tf : str, datetime.datetime
                End of forecast period (default is end of model analysis period).
            recalculate : bool
                Flag indicating forecast should be recalculated, otherwise forecast will be
                loaded from previous save file (if it exists).
            use_model : None or str
                Optionally pass path to pre-trained model directory in 'models'.
            n_jobs : int
                Number of cores to use.

            Returns:
            --------
            consensus : pd.DataFrame
                The model consensus, indexed by window date.
        """
        self._use_model = use_model
        makedir(self.preddir)

        # 
        self.ti_forecast = self.ti_model if ti is None else datetimeify(ti)
        self.tf_forecast = self.tf_model if tf is None else datetimeify(tf)
        if self.tf_forecast > self.data.tf:
            self.tf_forecast = self.data.tf
        if self.ti_forecast - self.dtw < self.data.ti:
            self.ti_forecast = self.data.ti+self.dtw

        loadFeatureMatrix = True

        model_path = self.modeldir + os.sep
        if use_model is not None:
            self._detect_model()
            model_path = self._use_model+os.sep
            
        model,classifier = get_classifier(self.classifier)

        # logic to determine which models need to be run and which to be 
        # read from disk
        pref = type(model).__name__
        fls = glob('{:s}/{:s}_*.pkl'.format(model_path, pref))
        load_predictions = []
        run_predictions = []
        if recalculate:
            run_predictions = fls
        else:
            for fl in fls:
                num = fl.split(os.sep)[-1].split('_')[-1].split('.')[0]
                flp = '{:s}/{:s}_{:s}.csv'.format(self.preddir, pref, num)
                if not os.path.isfile(flp):
                    run_predictions.append(flp)
                else:
                    load_predictions.append(flp)

        ys = []            
        # load existing predictions
        for fl in load_predictions:
            y = pd.read_csv(fl, index_col=0, parse_dates=['time'], infer_datetime_format=True)
            ys.append(y)

        # generate new predictions
        if len(run_predictions)>0:
            run_predictions = [(rp, rp.replace(model_path, self.preddir+os.sep).replace('.pkl','.csv')) for rp in run_predictions]

            # load feature matrix
            fM,_ = self._extract_features(self.ti_forecast, self.tf_forecast)

            # setup predictor
            if self.n_jobs > 1:
                p = Pool(self.n_jobs)
                mapper = p.imap
            else:
                mapper = map
            f = partial(predict_one_model, fM, model_path, pref)

            # train models with glorious progress bar
            for i, y in enumerate(mapper(f, run_predictions)):
                cf = (i+1)/len(run_predictions)
                print(f'forecasting: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
                ys.append(y)
            
            if self.n_jobs > 1:
                p.close()
                p.join()
        
        # condense data frames and write output
        ys = pd.concat(ys, axis=1, sort=False)
        consensus = np.mean([ys[col].values for col in ys.columns if 'pred' in col], axis=0)
        forecast = pd.DataFrame(consensus, columns=['consensus'], index=ys.index)

        ob_folder = os.path.join(self.consensusdir, self.od)
        wl_lfl_folder = os.path.join(ob_folder, f"{self.window}_{self.look_forward}")
        makedir(wl_lfl_folder)
        save_path = os.path.join(wl_lfl_folder, f"{cv}_consensus.csv")
        forecast.to_csv(save_path, index=True, index_label='time')
        #forecast.to_csv('{:s}/consensus.csv'.format(self.preddir), index=True, index_label='time')
        #forecast.to_csv('/Users/minoluke/Desktop/修士/test/tmp/consensus.csv'.format(self.preddir), index=True, index_label='time')
        # memory management
        if len(run_predictions)>0:
            del fM
            gc.collect()

        return forecast


def get_classifier(classifier):
    """ Return scikit-learn ML classifiers and search grids for input strings.

        Parameters:
        -----------
        classifier : str
            String designating which classifier to return.

        Returns:
        --------
        model : 
            Scikit-learn classifier object.
        grid : dict
            Scikit-learn hyperparameter grid dictionarie.

        Classifier options:
        -------------------
        DT - Decision Tree
    """

    if classifier == "DT":        # decision tree
        model = DecisionTreeClassifier(class_weight='balanced')
        grid = {'max_depth': [3,5,7], 'criterion': ['gini','entropy'],
            'max_features': ['auto','sqrt','log2',None]}
    else:
        raise ValueError("classifier '{:s}' not recognised".format(classifier))
    
    return model, grid


def train_one_model(fM, ys, Nfts, modeldir, classifier, retrain, random_seed, random_state):
    # undersample data
    rus = RandomUnderSampler(0.75, random_state=random_state+random_seed)
    fMt,yst = rus.fit_resample(fM,ys)
    yst = pd.Series(yst, index=range(len(yst)))
    fMt.index = yst.index

    # find significant features
    select = FeatureSelector(n_jobs=0, ml_task='classification')
    select.fit_transform(fMt,yst)
    fts = select.features[:Nfts]
    pvs = select.p_values[:Nfts]
    fMt = fMt[fts]
    with open('{:s}/{:04d}.fts'.format(modeldir, random_state),'w') as fp:
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))

    # get sklearn training objects
    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=random_state+random_seed)
    model, grid = get_classifier(classifier)            
        
    # check if model has already been trained
    pref = type(model).__name__
    fl = '{:s}/{:s}_{:04d}.pkl'.format(modeldir, pref, random_state)
    if os.path.isfile(fl) and not retrain:
        return
    
    # train and save classifier
    model_cv = GridSearchCV(model, grid, cv=ss, scoring="balanced_accuracy",error_score=np.nan)
    model_cv.fit(fMt,yst)
    _ = joblib.dump(model_cv.best_estimator_, fl, compress=3)

def predict_one_model(fM, model_path, pref, flp):
    flp,fl = flp
    num = flp.split(os.sep)[-1].split('.')[0].split('_')[-1]
    model = joblib.load(flp)
    with open(model_path+'{:s}.fts'.format(num)) as fp:
        lns = fp.readlines()
    fts = [' '.join(ln.rstrip().split()[1:]) for ln in lns]            
    
    # simulate predicton period
    yp = model.predict(fM[fts])
    
    # save prediction
    ypdf = pd.DataFrame(yp, columns=['pred{:s}'.format(num)], index=fM.index)
    ypdf.to_csv(fl, index=True, index_label='time')
    return ypdf

def datetimeify(t):
    """ Return datetime object corresponding to input string.

        Parameters:
        -----------
        t : str, datetime.datetime
            Date string to convert to datetime object.

        Returns:
        --------
        datetime : datetime.datetime
            Datetime object corresponding to input string.

        Notes:
        ------
        This function tries several datetime string formats, and raises a ValueError if none work.
    """
    if type(t) in [datetime, Timestamp]:
        return t
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S',]
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))


