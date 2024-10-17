# methods/test/test.py

import os
import pandas as pd
import numpy as np
import joblib
from glob import glob
from functools import partial
from methods.helper.helper import get_classifier


def forecast(fM, modeldir, preddir, n_jobs=1):
    """Use classifier models to forecast eruption likelihood.

    Parameters:
    -----------
    fM : pd.DataFrame
        Feature matrix.
    modeldir : str
        Directory containing trained models.
    preddir : str
        Directory to save predictions.
    n_jobs : int
        Number of jobs for parallel processing.

    Returns:
    --------
    forecast : pd.DataFrame
        Model consensus forecast.
    """
    # Logic to generate forecasts
    if n_jobs > 1:
        from multiprocessing import Pool

        p = Pool(n_jobs)
        mapper = p.imap
    else:
        mapper = map

    model_files = glob(os.path.join(modeldir, "*.pkl"))
    predictions = []

    for model_file in model_files:
        pref = os.path.basename(model_file).split("_")[0]
        prediction_file = os.path.join(
            preddir, os.path.basename(model_file).replace(".pkl", ".csv")
        )
        if os.path.isfile(prediction_file):
            y = pd.read_csv(prediction_file, index_col=0, parse_dates=["time"])
            predictions.append(y)
        else:
            f = partial(predict_one_model, fM, model_file, pref, prediction_file)
            y = list(mapper(f, [None]))[0]
            predictions.append(y)

    if n_jobs > 1:
        p.close()
        p.join()

    # Compute consensus
    ys = pd.concat(predictions, axis=1)
    consensus = np.mean([ys[col].values for col in ys.columns if "pred" in col], axis=0)
    forecast_df = pd.DataFrame(consensus, columns=["consensus"], index=ys.index)
    return forecast_df


def predict_one_model(fM, model_file, pref, prediction_file, _):
    """Predict using one model.

    Parameters:
    -----------
    fM : pd.DataFrame
        Feature matrix.
    model_file : str
        Path to the model file.
    pref : str
        Model prefix.
    prediction_file : str
        Path to save prediction.

    Returns:
    --------
    ypdf : pd.DataFrame
        Prediction dataframe.
    """
    num = os.path.basename(prediction_file).split(".")[0].split("_")[-1]
    model = joblib.load(model_file)
    fts_file = model_file.replace(".pkl", ".fts")
    with open(fts_file) as fp:
        lns = fp.readlines()
    fts = [" ".join(ln.rstrip().split()[1:]) for ln in lns]

    # Predict
    yp = model.predict(fM[fts])

    # Save prediction
    ypdf = pd.DataFrame(yp, columns=["pred{:s}".format(num)], index=fM.index)
    ypdf.to_csv(prediction_file, index=True, index_label="time")
    return ypdf


def detect_model(modeldir, classifier):
    """Detect trained models.

    Parameters:
    -----------
    modeldir : str
        Directory containing models.
    classifier : str
        Classifier type.

    Returns:
    --------
    model_list : list
        List of model file paths.
    """
    model_files = glob(os.path.join(modeldir, "{}_*.pkl".format(classifier)))
    return model_files
