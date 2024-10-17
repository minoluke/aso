#!/usr/bin/env python3

import os
import sys
import argparse
from datetime import timedelta
from methods.data.data import TremorData
from methods.train.train import train
from methods.test.test import forecast
from methods.helper.helper import datetimeify

# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
import warnings
from sklearn.exceptions import FitFailedWarning

logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


def forecast_dec_1day(od, wl, lf, cv):
    """Forecast model for Dec 2019 eruption"""

    # Constants
    month = timedelta(days=365.25 / 12)
    td = TremorData()

    # Define data streams based on 'od' argument
    data_streams_dict = {
        'tremor': ['rsam', 'mf', 'hf', 'dsar'],
        'gas': ['gas_max', 'gas_min', 'gas_mean', 'gas_number'],
        'magnetic': ['magnetic'],
        'kakou': ['kakouwall_temp'],
        'tilt': ['tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW'],
        'yudamari': ['yudamari_number', 'yudamari_temp'],
        'mid': ['rsam', 'mf', 'hf', 'dsar', 'gas_max', 'gas_min', 'gas_mean', 'gas_number', 'magnetic', 'yudamari_number', 'yudamari_temp'],
        'long': ['magnetic', 'tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW', 'yudamari_number', 'yudamari_temp', 'kakouwall_temp', 'gas_max', 'gas_min', 'gas_mean', 'gas_number'],
        'all': ['rsam', 'mf', 'hf', 'dsar', 'gas_max', 'gas_min', 'gas_mean', 'gas_number', 'magnetic', 'kakouwall_temp', 'tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW', 'yudamari_number', 'yudamari_temp']
    }

    if od not in data_streams_dict:
        raise ValueError(f"Invalid value for 'od': {od}")

    data_streams = data_streams_dict[od]

    # Construct ForecastModel object
    ti = '2010-01-01'
    tf = '2022-12-31'
    window = float(wl)
    look_forward = float(lf)
    overlap = 0.85
    n_jobs = 6  # Number of CPUs to use for parallel tasks

    # Columns to drop from feature matrix due to high correlation


    # Get the eruption event time for cross-validation (cv)
    te = td.tes[int(cv)]

    # Train the forecast model
    train_args = {
        'cv': cv,
        'ti': ti,
        'tf': tf,
        'retrain': True,
        'exclude_dates': [[te - 6 * month, te + 6 * month]],
        'n_jobs': n_jobs
    }

    # Perform training
    fm = train(
        window=window,
        overlap=overlap,
        look_forward=look_forward,
        data_streams=data_streams,
        **train_args
    )

    # Perform forecast
    forecast_args = {
        'cv': cv,
        'ti': ti,
        'tf': tf,
        'recalculate': True,
        'n_jobs': n_jobs
    }

    forecast_results = forecast(fm, **forecast_args)
    return forecast_results


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Forecast Dec 2019 eruption using seismic data.')
    parser.add_argument('od', type=str, help='Observation data type (e.g., tremor, gas, magnetic, all)')
    parser.add_argument('wl', type=str, help='Window length (e.g., 1.0)')
    parser.add_argument('lf', type=str, help='Look forward length (e.g., 1.0)')
    parser.add_argument('cv', type=str, help='Cross-validation index (e.g., 0)')

    args = parser.parse_args()

    # Call forecast function
    forecast_dec_1day(args.od, args.wl, args.lf, args.cv)
