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
    else:
        raise ValueError("Invalid value for 'od'")

    fm = ForecastModel(ti='2010-01-01', tf='2022-12-31', window=float(wl), overlap=0.75, look_forward=float(lf), data_streams=data_streams, od=od)
    
    # columns to manually drop from feature matrix because they are highly correlated to other 
    # linear regressors
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6

    # train the model, excluding 2019 eruption
    # note: building the feature matrix may take several hours, but only has to be done once 
    # and will intermittantly save progress in ../features/
    # trained scikit-learn models will be saved to ../models/*root*/
    te = td.tes[int(cv)]
    fm.train(ti='2010-01-01', tf='2022-12-31', drop_features=drop_features, retrain=True, exclude_dates=[[te-6*month,te+6*month],], n_jobs=n_jobs)      
    #fm.train(ti='2010-01-01', tf='2018-12-31', drop_features=drop_features, retrain=True, n_jobs=n_jobs) 
    # run forecast from 2011 to 2020
    # model predictions will be saved to ../predictions/*root*/ 
    
    #ys = fm.forecast(ti='2019-01-01', tf='2022-12-31', recalculate=True, n_jobs=n_jobs)    
    ys = fm.forecast(cv=cv, ti='2010-01-01', tf='2022-12-31', recalculate=True, n_jobs=n_jobs)  

    # plot forecast and quality metrics
    # plots will be saved to ../plots/*root*/
    #fm.plot_forecast(ys, threshold=0.8, xlim = [te-month/4., te+month/15.], 
    #    save=r'{:s}/forecast.png'.format(fm.plotdir))
    #fm.plot_accuracy(ys, save=r'{:s}/accuracy.png'.format(fm.plotdir))

    # construct a high resolution forecast (10 min updates) around the Dec 2019 eruption
    # note: building the feature matrix might take a while
    #fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=te+month/30, recalculate=True,save=r'{:s}/forecast_hires.png'.format(fm.plotdir), n_jobs=n_jobs)

def forecast_test():
    ''' test scale forecast model
    '''
    # constants
    month = timedelta(days=365.25/12)
        
    # set up model
    data_streams = [
    'magnetic',
    'gas_max',
    'gas_min',
    'gas_mean',
    'gas_number',
    'yudamari_number',
    'yudamari_temp',
    'kakouwall_temp',
    'tilt1_NS',
    'tilt1_EW',
    'tilt2_NS',
    'tilt2_EW'
    ]
    fm = ForecastModel(ti='2014-04-01', tf='2014-10-01', window=30., overlap=0.75, 
        look_forward=5., data_streams=data_streams, root='test')
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 4
    
    # train the model
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2014-04-01', tf='2014-10-01', drop_features=drop_features, retrain=True,
        n_jobs=n_jobs)      

    # plot a forecast for a future eruption
    te = fm.data.tes[1]
    fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=te+month/30, recalculate=True, 
        save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

def forecast_now():
    ''' forecast model for present day 
    '''
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
        
    # pull the latest data from GeoNet
    td = TremorData()
    td.update()

    # model from 2011 to present day (td.tf)
    data_streams = [
    'magnetic',
    'gas_max',
    'gas_min',
    'gas_mean',
    'gas_number',
    'yudamari_number',
    'yudamari_temp',
    'kakouwall_temp',
    'tilt1_NS',
    'tilt1_EW',
    'tilt2_NS',
    'tilt2_EW'
    ]
    fm = ForecastModel(ti='2011-01-01', tf=td.tf, window=30, overlap=0.75,  
        look_forward=5, data_streams=data_streams, root='online_forecaster')
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6
    
    # The online forecaster is trained using all eruptions in the dataset. It only
    # needs to be trained once, or again after a new eruption.
    # (Hint: feature matrices can be copied from other models to avoid long recalculations
    # providing they have the same window length and data streams. Copy and rename 
    # to *root*_features.csv)
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, 
        retrain=False, n_jobs=n_jobs)      
    
    # forecast the last 7 days at high resolution
    fm.hires_forecast(ti=fm.data.tf - 7*day, tf=fm.data.tf, recalculate=True, 
        save='current_forecast.png', nztimezone=True, n_jobs=n_jobs)  

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
    #forecast_test()
    #forecast_now()
    