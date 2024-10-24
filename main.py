#!/usr/bin/env python3
import os
import sys
import argparse
import shutil

# Ensure the current directory is in the Python path
sys.path.insert(0, os.path.abspath('.'))

from methods import ForecastModel, TremorData
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
    """
    Function to train and forecast eruption likelihood based on the provided parameters.
    
    Parameters:
    -----------
    od : str
        Observation type (e.g., tremor, gas, magnetic, etc.).
    wl : float
        Window length in days.
    lf : float
        Look-forward period in days.
    cv : int
        Cross-validation fold index.
    """
    # Define time deltas
    month = timedelta(days=365.25 / 12)
    day = timedelta(days=1)
    
    # Initialize TremorData
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
        raise ValueError(f"Invalid observation type '{od}'. Choose from ['tremor', 'gas', 'magnetic', 'kakou', 'tilt', 'yudamari', 'all'].")
    
    # Initialize ForecastModel with the provided parameters
    fm = ForecastModel(
        window=wl,
        overlap=0.85,
        look_forward=lf,
        data_streams=data_streams,
        od=od  # Pass the 'od' parameter
        # n_jobs is handled internally; no need to pass here
    )

    # Retrieve the eruption event for the given cv
    try:
        te = td.tes[int(cv)]
    except IndexError:
        raise ValueError(f"No eruption event found for cv index {cv}")

    # Define exclusion period around the eruption event
    exclusion_start = te - 6 * month
    exclusion_end = te + 6 * month

    # Train the model with exclusion of dates around the eruption event
    fm.train(
        ti='2010-01-01',
        tf='2022-12-31',
        retrain=True,
        exclude_dates_ranges=[[exclusion_start, exclusion_end]],
        Nfts=20,          # Number of features to select
        Ncl=100,          # Number of classifiers to train
        classifier="DT",  # Classifier type
        random_seed=0,    # Seed for reproducibility
        n_jobs=6          # Number of parallel jobs
    )

    # Forecast using the trained models
    ys = fm.forecast(
        cv=cv,
        ti='2010-01-01',
        tf='2022-12-31',
        recalculate=False,
        use_model=None,    # Use the default model directory
        n_jobs=6
    )
    print(f"Forecast for CV {cv} completed and saved.")


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
        raise ValueError(f"Invalid 'od' parameter '{args.od}'. Choose from {valid_od}")

    # Call the forecasting function
    forecast_dec_1day(args.od, args.wl, args.lf, args.cv)
