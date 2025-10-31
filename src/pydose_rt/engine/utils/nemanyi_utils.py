import scipy.stats as stat
import numpy as np


__all__ = [
    "multivariate_normal",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "F_score",
    "fleiss_kappa",
    "r2_score",
    "compute_ranks",
    "nemenyi_test",
]


def r2_score(y_true, y_pred):
    """R squared (coefficient of determination) regression score function.
    Best possible score is 1.0, lower values are worse.
    Parameters
    ----------
    y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.
    y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.
    Returns
    -------
    z : float
        The R^2 score.
    Notes
    -----
    This is not a symmetric function.
    Unlike most other scores, R^2 score may be negative (it need not actually
    be the square of a quantity R).
    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <http://en.wikipedia.org/wiki/Coefficient_of_determination>`_
    Examples
    --------
    >>> from parsimony.utils.stats import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.948...
    """
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        else:
            return 0.0

    return 1 - numerator / denominator


def _critical_nemenyi_value(p_value, num_models):
    """Critical values for the Nemenyi test.
    Table obtained from: https://gist.github.com/garydoranjr/5016455
    """
    values = [  # p   0.01   0.05   0.10    Models
        [2.576, 1.960, 1.645],  # 2
        [2.913, 2.344, 2.052],  # 3
        [3.113, 2.569, 2.291],  # 4
        [3.255, 2.728, 2.460],  # 5
        [3.364, 2.850, 2.589],  # 6
        [3.452, 2.948, 2.693],  # 7
        [3.526, 3.031, 2.780],  # 8
        [3.590, 3.102, 2.855],  # 9
        [3.646, 3.164, 2.920],  # 10
        [3.696, 3.219, 2.978],  # 11
        [3.741, 3.268, 3.030],  # 12
        [3.781, 3.313, 3.077],  # 13
        [3.818, 3.354, 3.120],  # 14
        [3.853, 3.391, 3.159],  # 15
        [3.884, 3.426, 3.196],  # 16
        [3.914, 3.458, 3.230],  # 17
        [3.941, 3.489, 3.261],  # 18
        [3.967, 3.517, 3.291],  # 19
        [3.992, 3.544, 3.319],  # 20
        [4.015, 3.569, 3.346],  # 21
        [4.037, 3.593, 3.371],  # 22
        [4.057, 3.616, 3.394],  # 23
        [4.077, 3.637, 3.417],  # 24
        [4.096, 3.658, 3.439],  # 25
        [4.114, 3.678, 3.459],  # 26
        [4.132, 3.696, 3.479],  # 27
        [4.148, 3.714, 3.498],  # 28
        [4.164, 3.732, 3.516],  # 29
        [4.179, 3.749, 3.533],  # 30
        [4.194, 3.765, 3.550],  # 31
        [4.208, 3.780, 3.567],  # 32
        [4.222, 3.795, 3.582],  # 33
        [4.236, 3.810, 3.597],  # 34
        [4.249, 3.824, 3.612],  # 35
        [4.261, 3.837, 3.626],  # 36
        [4.273, 3.850, 3.640],  # 37
        [4.285, 3.863, 3.653],  # 38
        [4.296, 3.876, 3.666],  # 39
        [4.307, 3.888, 3.679],  # 40
        [4.318, 3.899, 3.691],  # 41
        [4.329, 3.911, 3.703],  # 42
        [4.339, 3.922, 3.714],  # 43
        [4.349, 3.933, 3.726],  # 44
        [4.359, 3.943, 3.737],  # 45
        [4.368, 3.954, 3.747],  # 46
        [4.378, 3.964, 3.758],  # 47
        [4.387, 3.973, 3.768],  # 48
        [4.395, 3.983, 3.778],  # 49
        [4.404, 3.992, 3.788],  # 50
    ]

    if num_models < 2 or num_models > 50:
        raise ValueError("num_models must be in [2, 50].")

    if p_value == 0.01:
        return values[num_models - 2][0]
    elif p_value == 0.05:
        return values[num_models - 2][1]
    elif p_value == 0.10:
        return values[num_models - 2][2]
    else:
        raise ValueError("p_value must be in {0.01, 0.05, 0.10}")


def compute_ranks(X, method="average"):
    """Assign ranks to data, dealing with ties appropriately.
    Uses scipy.stats.rankdata to compute the ranks of each row of the matrix X.
    Parameters
    ----------
    X : numpy array
        Computes the ranks of the rows of X.
    method : str
        The method used to assign ranks to tied elements. Must be one of
        "average", "min", "max", "dense" and "ordinal".
    Returns
    -------
    R : numpy array
        A matrix with the ranks computed from X. Has the same shape as X. Ranks
        begin at 1.
    """
    if method not in ["average", "min", "max", "dense", "ordinal"]:
        raise ValueError(
            'Method must be one of "average", "min", "max", ' '"dense" and "ordinal".'
        )

    n = X.shape[0]
    R = np.zeros(X.shape)
    for i in range(n):
        r = stat.rankdata(X[i, :], method=method)
        R[i, :] = r

    return R


def nemenyi_test(X, p_value=0.05, return_ranks=False, return_critval=False):
    """Performs the Nemenyi test for comparing a set of classifiers to each
    other.
    Parameters
    ----------
    X : numpy array of shape (num_datasets, num_models)
        The scores of the num_datasets datasets for each of the num_models
        models. X must have at least one row and between 2 and 50 columns.
    p_value : float
        The p-value of the test. Must be one of 0.01, 0.05 or 0.1. Default is
        p_value=0.05.
    return_ranks : bool
        Whether or not to return the computed ranks. Default is False, do not
        return the ranks.
    return_critval : bool
        Whether or not to return the computed critical value. Default is False,
        do not return the critical value.
    """
    num_datasets, num_models = X.shape
    R = compute_ranks(X)
    crit_val = _critical_nemenyi_value(p_value, num_models)
    CD = crit_val * np.sqrt(num_models * (num_models + 1) / (6.0 * num_datasets))

    sign = np.zeros((num_models, num_models))

    sign_bool = np.zeros((num_models, num_models))
    for j1 in range(num_models):
        for j2 in range(num_models):
            if np.abs(np.mean(R[:, j1] - R[:, j2])) > CD:
                sign[j1, j2] = 1
            else:
                sign[j1, j2] = 0
            if np.mean(R[:, j1] - R[:, j2]) > 0:
                sign_bool[j1, j2] = 1
            else:
                sign_bool[j1, j2] = -1

    if return_ranks:
        if return_critval:
            return sign, R, CD, sign_bool
        else:
            return sign, R, sign_bool
    else:
        if return_critval:
            return sign, CD, sign_bool
        else:
            return sign, sign_bool
