#!/usr/bin/env python3
import os, sys
import argparse
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel
from datetime import timedelta, datetime

# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


def forecast_dec_1day(od,wl,lf,cv):
    ''' forecast model for Dec 2019 eruption
    '''
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    td = TremorData()
        
    # construct model object
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
    elif od == 'mid':
        data_streams = ['rsam', 'mf', 'hf', 'dsar','gas_max', 'gas_min', 'gas_mean', 'gas_number','kakouwall_temp']
    elif od == 'long':
        data_streams = ['magnetic','tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW','yudamari_number', 'yudamari_temp']
    elif od == 'all':
        data_streams = ['rsam', 'mf', 'hf', 'dsar','gas_max', 'gas_min', 'gas_mean', 'gas_number','magnetic','kakouwall_temp','tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW','yudamari_number', 'yudamari_temp']
    else:
        raise ValueError("Invalid value for 'od'")

    fm = ForecastModel(ti='2010-01-01', tf='2022-12-31', window=float(wl), overlap=0.99, look_forward=float(lf), data_streams=data_streams, od=od)
    
    # columns to manually drop from feature matrix because they are highly correlated to other 
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6

    te = td.tes[int(cv)]
    fm.train(cv=cv, ti='2010-01-01', tf='2022-12-31', drop_features=drop_features, retrain=True, exclude_dates=[[te-6*month,te+6*month],], n_jobs=n_jobs)      
    #fm.train(ti='2010-01-01', tf='2018-12-31', drop_features=drop_features, retrain=True, n_jobs=n_jobs) 
    
    ys = fm.forecast(cv=cv, ti='2010-01-01', tf='2022-12-31', recalculate=True, n_jobs=n_jobs)  

if __name__ == "__main__":
    # 引数パーサーを作成
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # 引数を追加
    parser.add_argument('od', type=str, help='The od parameter')
    parser.add_argument('wl', type=str, help='The wl parameter')
    parser.add_argument('lf', type=str, help='The lf parameter')
    parser.add_argument('cv', type=str, help='The cv parameter')
    
    # 引数を解析
    args = parser.parse_args()
    
    # 関数を呼び出し
    forecast_dec_1day(args.od,args.wl, args.lf,args.cv)
    
