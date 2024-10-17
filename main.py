#!/usr/bin/env python3
import os
import sys
import argparse
import shutil

# Ensure the parent directory is in the Python path
sys.path.insert(0, os.path.abspath('.'))

from methods import TremorData, ForecastModel  # Updated import
from datetime import timedelta, datetime

# Suppress warnings from tsfresh and sklearn
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


def forecast_dec_1day(od, wl, lf, cv):
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    td = TremorData()

    # Determine data streams based on observation type
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
        data_streams = [
            'rsam', 'mf', 'hf', 'dsar',
            'gas_max', 'gas_min', 'gas_mean', 'gas_number',
            'magnetic', 'kakouwall_temp',
            'tilt1_NS', 'tilt1_EW', 'tilt2_NS', 'tilt2_EW',
            'yudamari_number', 'yudamari_temp'
        ]
    else:
        raise ValueError("Invalid value for 'od'")

    # Initialize ForecastModel with the provided parameters
    fm = ForecastModel(
        window=float(wl),
        overlap=0.85,
        look_forward=float(lf),
        data_streams=data_streams,
        ti='2010-01-01',
        tf='2022-12-31',
        od=od  # Pass the 'od' parameter
    )

    # Set the available CPUs higher or lower as appropriate
    n_jobs = 6

    # Retrieve the eruption event for the given cv
    try:
        te = td.tes[int(cv)]
    except IndexError:
        raise ValueError(f"No eruption event found for cv index {cv}")

    # Train the model with exclusion of dates around the eruption event
    fm.train(
        cv=cv,
        ti='2010-01-01',
        tf='2022-12-31',
        retrain=True,
        exclude_dates=[[te - 6*month, te + 6*month]],
        n_jobs=n_jobs
    )

    # Forecast using the trained models
    ys = fm.forecast(
        cv=cv,
        ti='2010-01-01',
        tf='2022-12-31',
        recalculate=True
    )


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run forecast model.')

    # Add arguments
    parser.add_argument('od', type=str, help='Observation type (e.g., tremor, gas, magnetic, etc.)')
    parser.add_argument('wl', type=float, help='Window length in days (e.g., 30.0)')
    parser.add_argument('lf', type=float, help='Look-forward period in days (e.g., 1.0)')
    parser.add_argument('cv', type=int, help='Cross-validation fold index (integer)')

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    valid_od = ['tremor', 'gas', 'magnetic', 'kakou', 'tilt', 'yudamari', 'all']
    if args.od not in valid_od:
        raise ValueError(f"Invalid 'od' parameter. Choose from {valid_od}")

    # Call the forecasting function
    forecast_dec_1day(args.od, args.wl, args.lf, args.cv)
