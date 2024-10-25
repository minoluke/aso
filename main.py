#!/usr/bin/env python3
import os, sys
import argparse
sys.path.insert(0, os.path.abspath('..'))
from modules import *
from datetime import timedelta
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


def forecast_dec_1day(od,lb,lf,cv):

    month = timedelta(days=365.25/12)
    td = ObservationData()
        
    if od == 'tremor':
        data_streams = ['rsam', 'mf', 'hf', 'dsar']
    elif od == 'gas':
        data_streams = ['gas_max', 'gas_min', 'gas_mean', 'gas_number']
    elif od == 'magnetic':
        data_streams = ['magnetic']
    elif od == 'kakou':
        data_streams = ['kakouwall_temp']
    elif od == 'tilt':
        data_streams = ['tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW']
    elif od == 'yudamari':
        data_streams = ['yudamari_number', 'yudamari_temp']
    elif od == 'all':
        data_streams = ['rsam', 'mf', 'hf', 'dsar','gas_max', 'gas_min', 'gas_mean', 'gas_number','magnetic','kakouwall_temp','tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW','yudamari_number', 'yudamari_temp']
    else:
        raise ValueError("Invalid value for 'od'")

    TrainModel = TrainModel(ti='2010-01-01', tf='2022-12-31', look_backward=float(lb), overlap=0.85, look_forward=float(lf), data_streams=data_streams, od=od)
    TestModel = TestModel(ti='2010-01-01', tf='2022-12-31', look_backward=float(lb), overlap=0.85, look_forward=float(lf), data_streams=data_streams, od=od)
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6

    te = td.tes[int(cv)]
    TrainModel.train(cv=cv, ti='2010-01-01', tf='2022-12-31', retrain=True, exclude_dates=[[te-6*month,te+6*month],], n_jobs=n_jobs)      
    
    TestModel.test(cv=cv, ti='2010-01-01', tf='2022-12-31', recalculate=True, n_jobs=n_jobs)  

if __name__ == "__main__":
    # 引数パーサーを作成
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # 引数を追加
    parser.add_argument('od', type=str, help='The observation data parameter')
    parser.add_argument('lb', type=str, help='The look backward parameter')
    parser.add_argument('lf', type=str, help='The look forward parameter')
    parser.add_argument('cv', type=str, help='The count volcanic eruption parameter')
    
    # 引数を解析
    args = parser.parse_args()
    
    # 関数を呼び出し
    forecast_dec_1day(args.od,args.lb, args.lf,args.cv)
    
