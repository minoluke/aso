#!/usr/bin/env python3
import os, sys, argparse, warnings, logging
from datetime import timedelta
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
from modules import *


# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be switched back on when debugging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)

from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')


data_streams_dict = {
        'tremor': ['rsam', 'mf', 'hf', 'dsar'],
        'gas': ['gas_max', 'gas_min', 'gas_mean', 'gas_number'],
        'magnetic': ['magnetic'],
        'kakou': ['kakouwall_temp'],
        'tilt': ['tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW'],
        'yudamari': ['yudamari_number', 'yudamari_temp'],
        'all': ['rsam', 'mf', 'hf', 'dsar', 'gas_max', 'gas_min', 'gas_mean', 'gas_number','magnetic', 'kakouwall_temp', 'tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW','yudamari_number', 'yudamari_temp']
    }

month = timedelta(days=365.25/12)
n_jobs = 6
observation_m = ObservationData()

start_period = '2010-01-01'
end_period = '2022-12-31'

overlap = 0.85
classifier = 'DT'
#all_classifiers = ['DT','XGBoost','LightGBM','CatBoost']

def one_train_test(od,lb,lf,cv):
    GPU_AVAILABLE = is_gpu_available()
    if GPU_AVAILABLE:
        print("GPU available")
    else:
        print("GPU not available")

    te = observation_m.tes[int(cv)]

    data_streams = data_streams_dict.get(od)
    if data_streams is None:
        raise ValueError("Invalid value for 'od'")
    
    train_m = TrainModel(ti=start_period, tf=end_period, look_backward=float(lb), overlap=overlap, look_forward=float(lf), data_streams=data_streams, od=od, cv=cv)
    train_m.train(cv=cv, ti=start_period, tf=end_period, retrain=True, exclude_dates=[[te-6*month,te+6*month],], n_jobs=n_jobs, classifier=classifier) 

    test_m = TestModel(ti=start_period, tf=end_period, look_backward=float(lb), overlap=overlap, look_forward=float(lf), data_streams=data_streams, od=od, cv=cv)
    test_m.test(cv=cv, ti=start_period, tf=end_period, recalculate=True, n_jobs=n_jobs, classifier=classifier)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('od', type=str, help='The observation data parameter')
    parser.add_argument('lb', type=str, help='The look backward parameter')
    parser.add_argument('lf', type=str, help='The look forward parameter')
    parser.add_argument('cv', type=str, help='The count volcanic eruption parameter')

    args = parser.parse_args()
    
    one_train_test(args.od,args.lb, args.lf,args.cv)
    
