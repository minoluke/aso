# methods/__init__.py

from .data import TremorData
from .train import train, exclude_dates_func, collect_features, load_data
from .test import forecast, predict_one_model, detect_model
from .helper import get_classifier, datetimeify, makedir
import os 
from datetime import datetime, timedelta


class ForecastModel:
    """ 
    Object for training and running forecast models.
    
    Attributes:
    -----------
    data : TremorData
        Object containing tremor data.
    window : float
        Length of data window in days.
    overlap : float
        Fraction of overlap between adjacent windows.
    look_forward : float
        Length of look-forward in days.
    modeldir : str
        Directory to save forecast models.
    featdir : str
        Directory to save feature matrices.
    featfile : str
        File path to save/load the feature matrix.
    preddir : str
        Directory to save forecast model predictions.
    consensusdir : str
        Directory to save consensus predictions.
    od : str
        Additional directory identifier for organizing outputs.
    n_jobs : int
        Number of CPUs to use for parallel tasks.
    """
    
    all_classifiers = ['DT']
    
    def __init__(self, window, overlap, look_forward, data_streams, root=None, od=None):
        self.window = window
        self.overlap = overlap
        self.look_forward = look_forward
        self.data_streams = data_streams
        self.data = TremorData()
        
        # Validate data streams
        if any([d not in self.data.df.columns for d in self.data_streams]):
            raise ValueError(f"Data restricted to any of {self.data.df.columns}")
        
        # Set default dates
        self.ti_model = self.data.ti
        self.tf_model = self.data.tf
        
        # Window calculations
        self.dtw = timedelta(days=self.window)
        self.dto = timedelta(days=(1.0 - self.overlap) * self.window)
        self.iw = int(self.window)
        self.io = int(self.overlap * self.iw)
        if self.io == self.iw:
            self.io -= 1
        
        self.overlap = float(self.io) / self.iw
        self.dto = timedelta(days=(1.0 - self.overlap) * self.window)
        
        self.exclude_dates_ranges = []
        self.update_feature_matrix = True
        self.n_jobs = 6
        
        # Naming convention and directories
        if root is None:
            self.root = f"fm_{self.window:.2f}wndw_{self.overlap:.2f}ovlp_{self.look_forward:.2f}lkfd_" + "_".join(sorted(self.data_streams))
        else:
            self.root = root
        self.rootdir = os.path.dirname(os.path.abspath(__file__))
        self.plotdir = os.path.join(self.rootdir, 'plots', self.root)
        self.modeldir = os.path.join(self.rootdir, 'models', self.root)
        self.featdir = os.path.join(self.rootdir, 'features')
        self.featfile = os.path.join(self.featdir, f"{self.root}_features.csv")
        self.preddir = os.path.join(self.rootdir, 'predictions', self.root)
        self.consensusdir = os.path.join(self.rootdir, 'consensus')
        self.od = od
    
    def train(self, ti=None, tf=None, Nfts=20, Ncl=100, retrain=False, classifier="DT", random_seed=0, n_jobs=6, exclude_dates_ranges=[]):
        """
        Construct and train classifier models.
        
        Parameters:
        -----------
        ti : str or datetime.datetime, optional
            Beginning of training period (default is beginning model analysis period).
        tf : str or datetime.datetime, optional
            End of training period (default is end of model analysis period).
        Nfts : int, optional
            Number of most-significant features to use in classifier. Default is 20.
        Ncl : int, optional
            Number of classifier models to train. Default is 100.
        retrain : bool, optional
            Use saved models (False) or train new ones. Default is False.
        classifier : str, optional
            String denoting which classifier to train (e.g., 'DT'). Default is 'DT'.
        random_seed : int, optional
            Seed for random operations to ensure reproducibility. Default is 0.
        n_jobs : int, optional
            Number of CPUs to use for parallel tasks. Default is 6.
        exclude_dates_ranges : list of lists, optional
            List of [start_date, end_date] pairs to exclude during training.
        """
        self.n_jobs = n_jobs
        train(
            data=self.data,
            modeldir=self.modeldir,
            featdir=self.featdir,
            featfile=self.featfile,
            window=self.window,
            overlap=self.overlap,
            look_forward=self.look_forward,
            data_streams=self.data_streams,  # Pass data_streams here
            ti=ti,
            tf=tf,
            Nfts=Nfts,
            Ncl=Ncl,
            retrain=retrain,
            classifier=classifier,
            random_seed=random_seed,
            n_jobs=n_jobs,
            exclude_dates_ranges=exclude_dates_ranges
        )
    
    def forecast(self, cv=0, ti=None, tf=None, recalculate=False, use_model=None, n_jobs=6):
        """
        Use classifier models to forecast eruption likelihood.
        
        Parameters:
        -----------
        cv : int, optional
            Cross-validation identifier. Default is 0.
        ti : str or datetime.datetime, optional
            Beginning of forecast period (default is beginning of model analysis period).
        tf : str or datetime.datetime, optional
            End of forecast period (default is end of model analysis period).
        recalculate : bool, optional
            Flag indicating forecast should be recalculated, otherwise forecast will be loaded from previous save file.
        use_model : str, optional
            Path to pre-trained model directory. If None, uses self.modeldir.
        n_jobs : int, optional
            Number of cores to use. Default is 6.
        
        Returns:
        --------
        forecast : pd.DataFrame
            The model consensus, indexed by window date.
        """
        return forecast(
            data=self.data,
            use_model=use_model if use_model else self.modeldir,
            ti=ti,
            tf=tf,
            recalculate=recalculate,
            n_jobs=n_jobs,
            rootdir=self.rootdir,
            preddir=self.preddir,
            consensusdir=self.consensusdir,
            window=self.window,
            look_forward=self.look_forward,
            overlap=self.overlap,
            od=self.od
        )