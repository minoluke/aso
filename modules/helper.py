from datetime import datetime
from pandas._libs.tslibs.timestamps import Timestamp
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    import cupy as cp
    from cuml import DecisionTreeClassifier as cuml_DecisionTreeClassifier
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0 
except ImportError:
    GPU_AVAILABLE = False

def datetimeify(t):
    """ Return datetime object corresponding to input string.

        Parameters:
        -----------
        t : str, datetime.datetime
            Date string to convert to datetime object.

        Returns:
        --------
        datetime : datetime.datetime
            Datetime object corresponding to input string.

        Notes:
        ------
        This function tries several datetime string formats, and raises a ValueError if none work.
    """
    if type(t) in [datetime, Timestamp]:
        return t
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S',]
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))

def get_classifier(classifier):
    """ Return scikit-learn ML classifiers and search grids for input strings.

        Parameters:
        -----------
        classifier : str
            String designating which classifier to return.

        Returns:
        --------
        model : 
            Scikit-learn classifier object.
        grid : dict
            Scikit-learn hyperparameter grid dictionarie.

        Classifier options:
        -------------------
        DT - Decision Tree
    """

    if classifier == "DT":  # Decision Tree
        if GPU_AVAILABLE:
            print("Using GPU Decision Tree")
            model = cuml_DecisionTreeClassifier(class_weight='balanced')
            grid = {
                'max_depth': [3, 5, 7],
                'criterion': ['gini'],  # cuMLは'gini'のみサポート
                'max_features': ['auto', 'sqrt', 'log2']
            }
        else:
            model = DecisionTreeClassifier(class_weight='balanced')
            grid = {
                'max_depth': [3, 5, 7],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'sqrt', 'log2', None]
            }
    elif classifier.lower() == 'xgboost':
        model = xgb.XGBClassifier(
            tree_method='gpu_hist',   
            gpu_id=0,                  
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss'
        )
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        return model, param_grid

    elif classifier.lower() == 'lightgbm':
        model = lgb.LGBMClassifier(
            device='gpu',             
            objective='binary',
            boosting_type='gbdt'
        )
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 63, 127]
        }
        return model, param_grid

    elif classifier.lower() == 'catboost':
        model = CatBoostClassifier(
            task_type='GPU',         
            devices='0',              
            verbose=0
        )
        param_grid = {
            'iterations': [100, 200],
            'depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5]
        }
        return model, param_grid
    else:
        raise ValueError(f"classifier '{classifier}' not recognised")
    
    return model, grid


