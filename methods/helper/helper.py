# methods/helper/helper.py

from sklearn.tree import DecisionTreeClassifier
from datetime import datetime


def get_classifier(classifier):
    """Return scikit-learn ML classifiers and search grids for input strings.

    Parameters:
    -----------
    classifier : str
        String designating which classifier to return.

    Returns:
    --------
    model :
        Scikit-learn classifier object.
    grid : dict
        Scikit-learn hyperparameter grid dictionaries.

    Classifier options:
    -------------------
    DT - Decision Tree
    """
    if classifier == "DT":
        model = DecisionTreeClassifier(class_weight="balanced")
        grid = {
            "max_depth": [3, 5, 7],
            "criterion": ["gini", "entropy"],
            "max_features": ["auto", "sqrt", "log2", None],
        }
    else:
        raise ValueError("classifier '{:s}' not recognised".format(classifier))

    return model, grid


def datetimeify(t):
    """Return datetime object corresponding to input string.

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
    if isinstance(t, datetime):
        return t
    fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y %m %d %H %M %S"]
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))