# methods/helper/helper.py

import os
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier

def get_classifier(classifier):
    """
    Return scikit-learn ML classifiers and search grids for input strings.

    Parameters:
    -----------
    classifier : str
        String designating which classifier to return.

    Returns:
    --------
    model : sklearn.base.BaseEstimator
        Scikit-learn classifier object.
    grid : dict
        Scikit-learn hyperparameter grid dictionary.

    Classifier options:
    -------------------
    DT - Decision Tree
    """

    if classifier == "DT":        # Decision Tree
        model = DecisionTreeClassifier(class_weight='balanced')
        grid = {
            'max_depth': [3, 5, 7],
            'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2', None]
        }
    else:
        raise ValueError(f"Classifier '{classifier}' not recognized. Available options: ['DT']")

    return model, grid


def datetimeify(t):
    """
    Return datetime object corresponding to input string.

    Parameters:
    -----------
    t : str or datetime.datetime
        Date string or datetime object to convert to datetime object.

    Returns:
    --------
    datetime : datetime.datetime
        Datetime object corresponding to input string.

    Notes:
    ------
    This function tries several datetime string formats, and raises a ValueError if none work.
    """
    if isinstance(t, datetime):
        return t
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S']
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except (ValueError, TypeError):
            pass
    raise ValueError(f"Time data '{t}' not in a recognized format.")


def makedir(name):
    """
    Create a directory if it does not exist.

    Parameters:
    -----------
    name : str
        Path of the directory to create.

    Returns:
    --------
    None
    """
    os.makedirs(name, exist_ok=True)
