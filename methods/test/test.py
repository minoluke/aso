# methods/test/test.py

import os
import shutil
import gc
import joblib
import numpy as np
import pandas as pd
from glob import glob
from functools import partial
from multiprocessing import Pool
from sklearn.model_selection import ShuffleSplit
from tsfresh.utilities.dataframe_functions import impute
from datetime import datetime, timedelta

from methods.helper.helper import get_classifier, datetimeify, makedir


def detect_model(use_model):
    """
    Checks whether and what models have already been run.

    Parameters:
    -----------
    use_model : str
        Path to the model directory to detect models.

    Returns:
    --------
    classifier : str
        The classifier type detected.
    
    Raises:
    -------
    ValueError:
        If no feature files are found or if model numbering is not consecutive.
        If models in the directory are not recognized.
    """
    fls = glob(os.path.join(use_model, '*.fts'))
    if len(fls) == 0:
        raise ValueError(f"No feature files in '{use_model}'")

    inds = [int(float(os.path.basename(fl).split('.')[0])) for fl in fls if ('all.fts' not in fl)]
    if not inds:
        raise ValueError(f"No valid feature files found in '{use_model}'")
    if max(inds) != (len(inds) - 1):
        raise ValueError(f"Feature file numbering in '{use_model}' appears not consecutive")
    
    all_classifiers = ['DT']  # This can be extended as needed
    for classifier in all_classifiers:
        model, _ = get_classifier(classifier)
        pref = type(model).__name__
        if all(os.path.isfile(os.path.join(use_model, f"{pref}_{ind:04d}.pkl")) for ind in inds):
            return classifier
    raise ValueError(f"Did not recognize models in '{use_model}'")


def predict_one_model(fM, model_path, pref, flp):
    """
    Predicts using a single trained model and saves the prediction.

    Parameters:
    -----------
    fM : pd.DataFrame
        Feature matrix.
    model_path : str
        Path to the directory containing the models.
    pref : str
        Prefix of the model files (e.g., 'DecisionTreeClassifier').
    flp : tuple
        Tuple containing (model_file_path, prediction_file_path).

    Returns:
    --------
    ypdf : pd.DataFrame
        DataFrame containing the prediction.
    """
    flp, fl = flp
    num = os.path.basename(flp).split('_')[-1].split('.')[0]
    model = joblib.load(flp)
    feature_file = os.path.join(model_path, f"{num}.fts")
    if not os.path.isfile(feature_file):
        raise FileNotFoundError(f"Feature file '{feature_file}' not found for model '{flp}'")
    
    with open(feature_file, 'r') as fp:
        lns = fp.readlines()
    fts = [' '.join(ln.rstrip().split()[1:]) for ln in lns]
    
    # Ensure that all required features are present
    missing_features = set(fts) - set(fM.columns)
    if missing_features:
        raise ValueError(f"Missing features for prediction: {missing_features}")
    
    # Perform prediction
    yp = model.predict(fM[fts])
    
    # Save prediction
    ypdf = pd.DataFrame(yp, columns=[f"pred{num}"], index=fM.index)
    ypdf.to_csv(fl, index=True, index_label='time')
    return ypdf


def forecast(data, use_model=None, ti=None, tf=None, recalculate=False, n_jobs=6, rootdir='save/test', preddir='predictions', consensusdir='consensus', window=30.0, look_forward=7.0, overlap=0.5, od=None, data_streams=None ):
    """
    Use classifier models to forecast eruption likelihood.

    Parameters:
    -----------
    data : TremorData
        Object containing tremor data.
    use_model : str, optional
        Path to pre-trained model directory. If None, uses self.modeldir.
    ti : str or datetime.datetime, optional
        Beginning of forecast period (default is beginning of model analysis period).
    tf : str or datetime.datetime, optional
        End of forecast period (default is end of model analysis period).
    recalculate : bool, optional
        Flag indicating forecast should be recalculated, otherwise forecast will be loaded from previous save file.
    n_jobs : int, optional
        Number of cores to use. Default is 6.
    rootdir : str, optional
        Root directory for saving predictions. Default is 'save/test'.
    preddir : str, optional
        Directory to save forecast model predictions. Default is 'predictions'.
    consensusdir : str, optional
        Directory to save consensus predictions. Default is 'consensus'.
    window : float, optional
        Length of data window in days. Default is 30.0.
    look_forward : float, optional
        Length of look-forward in days. Default is 7.0.
    overlap : float, optional
        Fraction of overlap between adjacent windows. Default is 0.5.
    od : str, optional
        Additional directory identifier for organizing outputs.

    Returns:
    --------
    forecast : pd.DataFrame
        The model consensus, indexed by window date.
    """
    # Set up directories
    if use_model is None:
        raise ValueError("use_model must be provided to specify the model directory.")
    makedir(preddir)
    
    # Initialize forecast period
    ti_forecast = datetimeify(ti) if ti else data.ti
    tf_forecast = datetimeify(tf) if tf else data.tf
    if tf_forecast > data.tf:
        tf_forecast = data.tf
    dtw = timedelta(days=window)
    dto = timedelta(days=(1.0 - overlap) * window)
    iw = int(window)
    io = int(overlap * iw)
    if io == iw:
        io -= 1
    
    if ti_forecast - dtw < data.ti:
        ti_forecast = data.ti + dtw
    
    # Detect classifier
    classifier = detect_model(use_model)
    model, _ = get_classifier(classifier)
    pref = type(model).__name__
    
    # Gather model files
    model_files = glob(os.path.join(use_model, f"{pref}_*.pkl"))
    if not model_files:
        raise ValueError(f"No models found in '{use_model}' with prefix '{pref}'")
    
    # Determine which predictions to run or load
    load_predictions = []
    run_predictions = []
    if recalculate:
        run_predictions = model_files
    else:
        for fl in model_files:
            num = os.path.basename(fl).split('_')[-1].split('.')[0]
            pred_file = os.path.join(preddir, f"{pref}_{num}.csv")
            if not os.path.isfile(pred_file):
                run_predictions.append(fl)
            else:
                load_predictions.append(pred_file)
    
    ys = []
    
    # Load existing predictions
    for fl in load_predictions:
        y = pd.read_csv(fl, index_col=0, parse_dates=['time'], infer_datetime_format=True)
        ys.append(y)
    
    # Generate new predictions
    if run_predictions:
        run_predictions = [(rp, os.path.join(preddir, f"{pref}_{os.path.basename(rp).split('_')[-1].replace('.pkl', '.csv')}")) for rp in run_predictions]
        
        # Load feature matrix
        from methods.feature_extract.feature_extract import _extract_features
        fm, _ = _extract_features(
            data=data,
            ti=ti_forecast,
            tf=tf_forecast,
            Nw=int(np.floor(((tf_forecast - ti_forecast) / timedelta(days=1)) / (iw - io))),
            iw=iw,
            io=io,
            dtw=dtw,
            dto=dto,
            data_streams=data_streams,
            featdir=os.path.join(rootdir, 'feature'),
            featfile=os.path.join(rootdir, 'feature', 'forecast_features.csv'),
            n_jobs=n_jobs,
            update_feature_matrix=True
        )
    
        # Set up multiprocessing pool
        if n_jobs > 1:
            pool = Pool(n_jobs)
            mapper = pool.imap
        else:
            mapper = map
    
        # Partial function for predict_one_model
        predict_func = partial(
            predict_one_model,
            fM=fm,
            model_path=use_model,
            pref=pref
        )
    
        # Run predictions with progress indication
        for i, y in enumerate(mapper(predict_func, run_predictions)):
            cf = (i + 1) / len(run_predictions)
            print(f"Forecasting: [{'#' * round(50 * cf) + '-' * round(50 * (1 - cf))}] {100. * cf:.2f}%", end='\r')
            ys.append(y)
        print("\nForecasting completed.")
    
        if n_jobs > 1:
            pool.close()
            pool.join()
    
    # Condense predictions and compute consensus
    if not ys:
        raise ValueError("No predictions were loaded or generated.")
    
    ys_concat = pd.concat(ys, axis=1, sort=False)
    consensus = ys_concat.mean(axis=1)
    forecast = pd.DataFrame(consensus, columns=['consensus'], index=ys_concat.index)
    
    # Save consensus prediction
    ob_folder = os.path.join(consensusdir, od if od else 'default')
    wl_lfl_folder = os.path.join(ob_folder, f"{window}_{look_forward}")
    makedir(wl_lfl_folder)
    save_path = os.path.join(wl_lfl_folder, f"consensus.csv")
    forecast.to_csv(save_path, index=True, index_label='time')
    print(f"Consensus forecast saved to {save_path}")
    
    # Memory management
    del fm, ys, ys_concat
    gc.collect()
    
    return forecast
