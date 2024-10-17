# methods/train/train.py

import os
import pandas as pd
import numpy as np
import joblib
from glob import glob
from functools import partial
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from imblearn.under_sampling import RandomUnderSampler
from tsfresh.transformers import FeatureSelector
from methods.helper.helper import get_classifier, datetimeify


def train(
    fM,
    ys,
    Nfts,
    modeldir,
    classifier,
    Ncl=100,
    retrain=False,
    random_seed=0,
    n_jobs=1,
    exclude_dates_list=[],
):
    """Train classifier models.

    Parameters:
    -----------
    fM : pd.DataFrame
        Feature matrix.
    ys : pd.Series
        Label vector.
    Nfts : int
        Number of features to select.
    modeldir : str
        Directory to save models.
    classifier : str
        Classifier type.
    Ncl : int
        Number of classifiers to train.
    retrain : bool
        Whether to retrain existing models.
    random_seed : int
        Random seed for reproducibility.
    n_jobs : int
        Number of jobs for parallel processing.
    """
    if n_jobs > 1:
        from multiprocessing import Pool

        p = Pool(n_jobs)
        mapper = p.imap
    else:
        mapper = map
    f = partial(
        train_one_model,
        fM,
        ys,
        Nfts,
        modeldir,
        classifier,
        retrain,
        random_seed,
    )
    for i, _ in enumerate(mapper(f, range(Ncl))):
        cf = (i + 1) / Ncl
        print(
            f'building models: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r',
            end="",
        )
    if n_jobs > 1:
        p.close()
        p.join()


def train_one_model(
    fM,
    ys,
    Nfts,
    modeldir,
    classifier,
    retrain,
    random_seed,
    random_state,
):
    """Train one classifier model.

    Parameters:
    -----------
    fM : pd.DataFrame
        Feature matrix.
    ys : pd.Series
        Label vector.
    Nfts : int
        Number of features to select.
    modeldir : str
        Directory to save model.
    classifier : str
        Classifier type.
    retrain : bool
        Whether to retrain existing models.
    random_seed : int
        Random seed.
    random_state : int
        Random state for reproducibility.
    """
    # Undersample data
    rus = RandomUnderSampler(0.75, random_state=random_state + random_seed)
    fMt, yst = rus.fit_resample(fM, ys)
    yst = pd.Series(yst, index=range(len(yst)))
    fMt.index = yst.index

    # Feature selection
    select = FeatureSelector(n_jobs=0, ml_task="classification")
    select.fit_transform(fMt, yst)
    fts = select.features[:Nfts]
    pvs = select.p_values[:Nfts]
    fMt = fMt[fts]
    with open("{:s}/{:04d}.fts".format(modeldir, random_state), "w") as fp:
        for f, pv in zip(fts, pvs):
            fp.write("{:4.3e} {:s}\n".format(pv, f))

    # Get classifier
    model, grid = get_classifier(classifier)

    # Check if model exists
    pref = type(model).__name__
    fl = "{:s}/{:s}_{:04d}.pkl".format(modeldir, pref, random_state)
    if os.path.isfile(fl) and not retrain:
        return

    # Train and save classifier
    ss = ShuffleSplit(
        n_splits=5, test_size=0.25, random_state=random_state + random_seed
    )
    model_cv = GridSearchCV(
        model, grid, cv=ss, scoring="balanced_accuracy", error_score=np.nan
    )
    model_cv.fit(fMt, yst)
    _ = joblib.dump(model_cv.best_estimator_, fl, compress=3)


def exclude_dates(X, y, exclude_dates_list):
    """Exclude specified date ranges from data.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Label vector.
    exclude_dates_list : list
        List of date ranges to exclude.

    Returns:
    --------
    Xr : pd.DataFrame
        Reduced feature matrix.
    yr : pd.Series
        Reduced label vector.
    """
    if len(exclude_dates_list) != 0:
        for exclude_date_range in exclude_dates_list:
            t0, t1 = [datetimeify(dt) for dt in exclude_date_range]
            inds = (y.index < t0) | (y.index >= t1)
            X = X.loc[inds]
            y = y.loc[inds]
    return X, y


def collect_features(modeldir, save=None):
    """Collect features used in trained classifiers.

    Parameters:
    -----------
    modeldir : str
        Directory containing model feature files.
    save : str or None
        Path to save collected features.

    Returns:
    --------
    labels : list
        List of feature names.
    freqs : list
        Frequencies of features in models.
    """
    fls = glob("{:s}/*.fts".format(modeldir))
    feats = []
    for fl in fls:
        if fl.endswith("all.fts") or fl.endswith("ranked.fts"):
            continue
        with open(fl) as fp:
            lns = fp.readlines()
        feats += [" ".join(ln.rstrip().split()[1:]) for ln in lns]

    labels = list(set(feats))
    freqs = [feats.count(label) for label in labels]
    labels = [label for _, label in sorted(zip(freqs, labels))][::-1]
    freqs = sorted(freqs)[::-1]
    if save is not None:
        with open(save, "w") as fp:
            _ = [fp.write("{:d},{:s}\n".format(freq, ft)) for freq, ft in zip(freqs, labels)]
    return labels, freqs


def load_data(featfile, ti, tf):
    """Load feature matrix and label vector.

    Parameters:
    -----------
    featfile : str
        Path to feature file.
    ti : datetime.datetime
        Start time.
    tf : datetime.datetime
        End time.

    Returns:
    --------
    fM : pd.DataFrame
        Feature matrix.
    ys : pd.Series
        Label vector.
    """
    # Implement code to load data
    pass
