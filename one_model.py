import os, joblib
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from tsfresh.transformers import FeatureSelector
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from helper import get_classifier

def train_one_model(fM, ys, Nfts, modeldir, classifier, retrain, random_seed, random_state):
    # undersample data
    rus = RandomUnderSampler(0.75, random_state=random_state+random_seed)
    fMt,yst = rus.fit_resample(fM,ys)
    yst = pd.Series(yst, index=range(len(yst)))
    fMt.index = yst.index

    # find significant features
    select = FeatureSelector(n_jobs=0, ml_task='classification')
    select.fit_transform(fMt,yst)
    fts = select.features[:Nfts]
    pvs = select.p_values[:Nfts]
    fMt = fMt[fts]
    with open('{:s}/{:04d}.fts'.format(modeldir, random_state),'w') as fp:
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))

    # get sklearn training objects
    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=random_state+random_seed)
    model, grid = get_classifier(classifier)            
        
    # check if model has already been trained
    pref = type(model).__name__
    fl = '{:s}/{:s}_{:04d}.pkl'.format(modeldir, pref, random_state)
    if os.path.isfile(fl) and not retrain:
        return
    
    # train and save classifier
    model_cv = GridSearchCV(model, grid, cv=ss, scoring="balanced_accuracy",error_score=np.nan)
    model_cv.fit(fMt,yst)
    _ = joblib.dump(model_cv.best_estimator_, fl, compress=3)

def predict_one_model(fM, model_path, pref, flp):
    flp,fl = flp
    num = flp.split(os.sep)[-1].split('.')[0].split('_')[-1]
    model = joblib.load(flp)
    with open(model_path+'{:s}.fts'.format(num)) as fp:
        lns = fp.readlines()
    fts = [' '.join(ln.rstrip().split()[1:]) for ln in lns]            
    
    # simulate predicton period
    yp = model.predict(fM[fts])
    
    # save prediction
    ypdf = pd.DataFrame(yp, columns=['pred{:s}'.format(num)], index=fM.index)
    ypdf.to_csv(fl, index=True, index_label='time')
    return ypdf