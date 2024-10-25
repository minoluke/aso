#!/usr/bin/env python3
import os, sys, argparse, logging, warnings
from datetime import timedelta
from sklearn.exceptions import FitFailedWarning
for warning in [UserWarning, FutureWarning, FitFailedWarning]:
    warnings.filterwarnings("ignore", category=warning)

sys.path.insert(0, os.path.abspath('..'))
from modules import *
logging.getLogger("tsfresh").setLevel(logging.ERROR)

import time 

def train_test(od,lb,lf,cv):
    start_time = time.time()

    month = timedelta(days=365.25/12)
    observation_m = ObservationData()
    te = observation_m.tes[int(cv)]

    n_jobs = 6
        
    data_streams_dict = {
        'tremor': ['rsam', 'mf', 'hf', 'dsar'],
        'gas': ['gas_max', 'gas_min', 'gas_mean', 'gas_number'],
        'magnetic': ['magnetic'],
        'kakou': ['kakouwall_temp'],
        'tilt': ['tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW'],
        'yudamari': ['yudamari_number', 'yudamari_temp'],
        'all': ['rsam', 'mf', 'hf', 'dsar', 'gas_max', 'gas_min', 'gas_mean', 'gas_number','magnetic', 'kakouwall_temp', 'tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW','yudamari_number', 'yudamari_temp']
    }

    data_streams = data_streams_dict.get(od)
    if data_streams is None:
        raise ValueError("Invalid value for 'od'")

    
    train_m = TrainModel(ti='2010-01-01', tf='2022-12-31', look_backward=float(lb), overlap=0.85, look_forward=float(lf), data_streams=data_streams, od=od)
    train_m.train(cv=cv, ti='2010-01-01', tf='2022-12-31', retrain=True, exclude_dates=[[te-6*month,te+6*month],], n_jobs=n_jobs) 

    test_start = time.time()
    test_m = TestModel(ti='2010-01-01', tf='2022-12-31', look_backward=float(lb), overlap=0.85, look_forward=float(lf), data_streams=data_streams, od=od)
    test_m.test(cv=cv, ti='2010-01-01', tf='2022-12-31', recalculate=True, n_jobs=n_jobs)  
    print(f"Testing time: {time.time() - test_start:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('od', type=str, help='The observation data parameter')
    parser.add_argument('lb', type=str, help='The look backward parameter')
    parser.add_argument('lf', type=str, help='The look forward parameter')
    parser.add_argument('cv', type=str, help='The count volcanic eruption parameter')

    args = parser.parse_args()
    
    train_test(args.od,args.lb, args.lf,args.cv)
    
