# methods/train/train.py

import os
import shutil
import gc
import joblib
import numpy as np
import pandas as pd
from glob import glob
from functools import partial
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from tsfresh.transformers import FeatureSelector
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from datetime import datetime, timedelta

from methods.helper.helper import get_classifier, makedir, datetimeify
from methods.feature_extract.feature_extract import _construct_windows, extract_features

def train_one_model(fM, ys, Nfts, modeldir, classifier, retrain, random_seed,random_state):
    """
    Train a single model with undersampling and feature selection.
    """
    # Undersample the data
    rus = RandomUnderSampler(sampling_strategy=0.75, random_state=random_state + random_seed)
    fMt, yst = rus.fit_resample(fM, ys)
    yst = pd.Series(yst, index=range(len(yst)))
    fMt.index = yst.index

    # Feature selection
    select = FeatureSelector(n_jobs=0, ml_task='classification')
    select.fit_transform(fMt, yst)
    fts = select.features_[:Nfts]
    pvs = select.p_values_[:Nfts]
    fMt = fMt[fts]

    # Save selected features and their p-values
    feature_file = os.path.join(modeldir, f"{random_state:04d}.fts")
    with open(feature_file, 'w') as fp:
        for f, pv in zip(fts, pvs):
            fp.write(f"{pv:4.3e} {f}\n")

    # Initialize classifier
    model, grid = get_classifier(classifier)
    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=random_state + random_seed)

    # Check if model already exists
    pref = type(model).__name__
    model_file = os.path.join(modeldir, f"{pref}_{random_state:04d}.pkl")
    if os.path.isfile(model_file) and not retrain:
        return

    # Train classifier with GridSearchCV
    model_cv = GridSearchCV(model, grid, cv=ss, scoring="balanced_accuracy", error_score=np.nan)
    model_cv.fit(fMt, yst)

    # Save the best estimator
    joblib.dump(model_cv.best_estimator_, model_file, compress=3)


def exclude_dates_func(X, y, exclude_dates_ranges):
    """
    Drop rows from feature matrix and label vector based on exclusion periods.
    """
    for exclude_dates_range in exclude_dates_ranges:
        t0, t1 = [datetimeify(dt) for dt in exclude_dates_range]
        inds = (y.index < t0) | (y.index >= t1)
        X = X.loc[inds]
        y = y.loc[inds]
    return X, y


def collect_features(modeldir, save=None):
    """
    Aggregate features used to train classifiers by frequency.
    """
    if save is None:
        save = os.path.join(modeldir, 'all.fts')

    feats = []
    fls = glob(os.path.join(modeldir, '*.fts'))
    for fl in fls:
        if os.path.basename(fl).split('.')[0] in ['all', 'ranked']:
            continue
        with open(fl, 'r') as fp:
            lns = fp.readlines()
        feats += [' '.join(ln.rstrip().split()[1:]) for ln in lns]

    labels = list(set(feats))
    freqs = [feats.count(label) for label in labels]
    sorted_indices = np.argsort(freqs)[::-1]
    labels = [labels[i] for i in sorted_indices]
    freqs = sorted(freqs, reverse=True)

    # Save feature frequencies
    with open(save, 'w') as fp:
        for freq, ft in zip(freqs, labels):
            fp.write(f"{freq},{ft}\n")

    return labels, freqs


def load_data(data, ti, tf, iw, io, dtw, dto, Nw, data_streams, featdir, featfile, n_jobs=6, update_feature_matrix=True):
    """
    Load feature matrix and label vector for a given period.
    """
    makedir(featdir)

    # Features to compute
    cfp = ComprehensiveFCParameters()

    # Check if feature matrix already exists and what it contains
    if os.path.isfile(featfile):
        existing_fm = pd.read_csv(featfile, index_col=0, parse_dates=['time'], infer_datetime_format=True)
        ti0, tf0 = existing_fm.index[0], existing_fm.index[-1]
        Nw0 = len(existing_fm)
        existing_features = set([hd.split('__')[1] for hd in existing_fm.columns])

        # Determine padding
        pad_left = int((ti0 - ti) / dto) if ti < ti0 else 0
        pad_right = int(((ti + (Nw - 1) * dto) - tf0) / dto) if tf > tf0 else 0
        i0 = abs(pad_left) if pad_left < 0 else 0
        i1 = Nw0 + max(pad_left, 0) + pad_right

        # Determine new features
        new_features = set(cfp.keys()) - existing_features
        more_cols = bool(new_features)
        if more_cols:
            cfp = {k: v for k, v in cfp.items() if k in new_features}

        # Update feature matrix if needed
        if (more_cols or pad_left > 0 or pad_right > 0) and update_feature_matrix:
            fm = existing_fm.copy()

            # Add new columns
            if more_cols:
                df_new, wd_new = _construct_windows(data, Nw0, ti0, iw, io, dtw, dto, data_streams, i0=0, i1=Nw0)
                fm_new = extract_features(df_new, column_id='id', n_jobs=n_jobs, default_fc_parameters=cfp, impute_function=impute)
                fm_new.index = pd.Series(wd_new)
                fm = pd.concat([fm, fm_new], axis=1, sort=False)

            # Add new rows on the left
            if pad_left > 0:
                df_left, wd_left = _construct_windows(data, pad_left, ti, iw, io, dtw, dto, data_streams, i0=0, i1=pad_left)
                fm_left = extract_features(df_left, column_id='id', n_jobs=n_jobs, default_fc_parameters=cfp, impute_function=impute)
                fm_left.index = pd.Series(wd_left)
                fm = pd.concat([fm_left, fm], sort=False)

            # Add new rows on the right
            if pad_right > 0:
                df_right, wd_right = _construct_windows(data, pad_right, ti + (Nw - pad_right) * dto, iw, io, dtw, dto, data_streams, i0=0, i1=pad_right)
                fm_right = extract_features(df_right, column_id='id', n_jobs=n_jobs, default_fc_parameters=cfp, impute_function=impute)
                fm_right.index = pd.Series(wd_right)
                fm = pd.concat([fm, fm_right], sort=False)

            # Save updated feature matrix
            fm.to_csv(featfile, index=True, index_label='time')
            fm = fm.iloc[i0:i1]
        else:
            # Read relevant part of the existing feature matrix
            fm = existing_fm.iloc[i0:i1]
    else:
        # Create feature matrix from scratch
        df_windows, wd = _construct_windows(data, Nw, ti, iw, io, dtw, dto, data_streams)
        fm = extract_features(df_windows, column_id='id', n_jobs=n_jobs, default_fc_parameters=cfp, impute_function=impute)
        fm.index = pd.Series(wd)
        fm.to_csv(featfile, index=True, index_label='time')

    # Compute labels
    ys = pd.DataFrame([data._is_eruption_in(days=dtw.days, from_time=t) for t in pd.to_datetime(fm.index)], columns=['label'], index=fm.index)
    return fm, ys


def train(data, modeldir, featdir, featfile, window, overlap, look_forward, data_streams, ti=None, tf=None,
          Nfts=20, Ncl=100, retrain=False, classifier="DT", random_seed=0, n_jobs=6, exclude_dates_ranges=[]):
    """
    Classifier モデルを構築・訓練する。
    
    Parameters:
    -----------
    data : TremorData
        Tremor データを含むオブジェクト。
    modeldir : str
        モデルを保存するディレクトリ。
    featdir : str
        特徴量マトリックスを保存するディレクトリ。
    featfile : str
        特徴量マトリックスのファイルパス。
    window : float
        ウィンドウの長さ（日数）。
    overlap : float
        ウィンドウの重なり率。
    look_forward : float
        予測期間（日数）。
    data_streams : list
        特徴量を抽出するデータストリーム。
    ti : str or datetime.datetime, optional
        訓練期間の開始時刻。指定がなければデータの開始時刻。
    tf : str or datetime.datetime, optional
        訓練期間の終了時刻。指定がなければデータの終了時刻。
    Nfts : int, optional
        使用する特徴量の数。デフォルトは20。
    Ncl : int, optional
        訓練する分類器モデルの数。デフォルトは100。
    retrain : bool, optional
        保存されたモデルを使用するか、新たに訓練するか。デフォルトはFalse。
    classifier : str, optional
        訓練する分類器の種類（例："DT"）。デフォルトは"DT"。
    random_seed : int, optional
        再現性のためのランダムシード。デフォルトは0。
    n_jobs : int, optional
        並列処理に使用するCPUコア数。デフォルトは6。
    exclude_date_ranges : list of lists, optional
        訓練中に除外する期間のリスト（各要素は [start_date, end_date] のリスト）。
    """
    makedir(modeldir)

    # 訓練期間の初期化
    ti_train = datetime.strptime(ti, '%Y-%m-%d') if isinstance(ti, str) else ti if ti else data.ti
    tf_train = datetime.strptime(tf, '%Y-%m-%d') if isinstance(tf, str) else tf if tf else data.tf

    if tf_train > data.tf:
        raise ValueError(f"Model end date '{tf_train}' beyond data range '{data.tf}'")
    if ti_train < data.ti:
        raise ValueError(f"Model start date '{ti_train}' predates data range '{data.ti}'")

    # ウィンドウパラメータの定義
    dtw = timedelta(days=window)
    dto = timedelta(days=(1.0 - overlap) * window)
    iw = int(window)
    io = int(overlap * iw)
    if io == iw:
        io -= 1

    overlap = float(io) / iw
    dto = timedelta(days=(1.0 - overlap) * window)

    # 訓練が必要かどうかのチェック
    if not retrain:
        run_models = False
        model, _ = get_classifier(classifier)
        pref = type(model).__name__
        for i in range(Ncl):
            model_file = os.path.join(modeldir, f"{pref}_{i:04d}.pkl")
            if not os.path.isfile(model_file):
                run_models = True
                break
        if not run_models:
            print("All models are already trained. Skipping training.")
            return
    else:
        # 既存のモデルファイルを削除
        old_models = glob(os.path.join(modeldir, '*'))
        for fl in old_models:
            os.remove(fl)
        print("Old model files removed.")

    # 特徴量マトリックスとラベルベクトルの読み込み
    fM, ys = load_data(
        data=data,
        ti=ti_train,
        tf=tf_train,
        iw=iw,
        io=io,
        dtw=dtw,
        dto=dto,
        Nw=int(np.floor(((tf_train - ti_train) / timedelta(days=1)) / (iw - io))),
        data_streams=data_streams,  # data_streams をここで渡す
        featdir=featdir,
        featfile=featfile,
        n_jobs=n_jobs,
        update_feature_matrix=True
    )

    # 指定された期間を除外
    X_filtered, y_filtered = exclude_dates_func(fM, ys['label'], exclude_dates_ranges)

    if y_filtered.shape[0] != X_filtered.shape[0]:
        raise ValueError("Dimensions of feature matrix and label vector do not match after excluding dates.")


    # モデル訓練のセットアップ
    if n_jobs > 1:
        pool = Pool(n_jobs)
        mapper = pool.starmap
    else:
        mapper = None  # シングルプロセスの場合は後で直接呼び出す

    # 引数のリストを準備
    args = [
        (X_filtered, y_filtered, Nfts, modeldir, classifier, retrain, random_seed, i)
        for i in range(Ncl)
    ]

    # モデルの訓練（進捗表示付き）
    if n_jobs > 1:
        # 並列処理
        results = mapper(train_one_model, args)
        for i, _ in enumerate(results):
            cf = (i + 1) / Ncl
            print(f"Building models: [{'#' * round(50 * cf) + '-' * round(50 * (1 - cf))}] {100. * cf:.2f}%", end='\r')
    else:
        # シングルプロセス
        for i, _ in enumerate([train_one_model(*arg) for arg in args]):
            cf = (i + 1) / Ncl
            print(f"Building models: [{'#' * round(50 * cf) + '-' * round(50 * (1 - cf))}] {100. * cf:.2f}%", end='\r')

    print("\nModel training completed.")

    if n_jobs > 1:
        pool.close()
        pool.join()

    # メモリの解放
    del fM, ys, X_filtered, y_filtered
    gc.collect()

    # 特徴量頻度の収集
    collect_features(modeldir)

    # 'all.fts' をコンセンサスディレクトリにコピー
    all_fts_path = os.path.join(modeldir, 'all.fts')
    if os.path.exists(all_fts_path):
        consensus_dir = os.path.join('save', 'consensus', f"{window}_{look_forward}")
        makedir(consensus_dir)
        new_all_fts_path = os.path.join(consensus_dir, f"cv_{os.path.basename(all_fts_path)}")
        shutil.copy(all_fts_path, new_all_fts_path)
        print(f"Feature frequencies copied to {new_all_fts_path}")