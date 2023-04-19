import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

rng = np.random.default_rng()


def load_config():
    file_path = 'config.yaml'
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def generate_weibull(n: int, data_cutoff: float, scale: float, shape: float, const=0, gauss_noise=True) -> pd.DataFrame:
    """_summary_

    Args:
        n (int): Number of patients
        data_cutoff (float): Data cutoff time in months
        scale (float): Weibull scale parameter
        shape (float): Weibull shape parameter
        const (flaot, optional): cure rate in percentage. Defaults to 0.
        gauss_noise (bool, optional):  If true, add gaussian noise to time to event. Defaults to True.

    Returns:
        pd.DataFrame: Data frame of Time and Survival
    """    

    const_n = int(n * const / 100)
    s = np.linspace(const + 100 / n, 100, n - const_n)
    t = scale * (-np.log((s - const) / (100 - const)))**(1 / shape)
    if gauss_noise:
        t = t + rng.normal(0, 0.5, s.size)
    # scale time to fit [0, data_cutoff]
    # if t > data_cutoff, convert to data_cutoff
    t = np.where(t > data_cutoff, data_cutoff, t)
    t = np.where(t < 0, 0, t)  # if t < 0, convert it to 0
    t = np.hstack((t, np.repeat(data_cutoff, const_n)))
    df = pd.DataFrame(
        {'Time': np.sort(t)[::-1], 'Survival': np.linspace(0, 100 - 100 / n, n)})
    return df


def fit_rho(a, b, rho, rng, ori_rho=None):
    """ Shuffle data of two sorted dataset to make two dataset to have a desired Spearman correlation.
    Note that a and b should be have the same length.
    Modified from: https://www.mathworks.com/help/stats/generate-correlated-data-using-rank-correlation.html

    Args:
        a (array_like): sorted dataset 1
        b (array_like): sorted dataset 2
        rho (float): desired spearman correlation coefficient
        ori_rho (float): internal argument for recursive part (default: None)
        seed (int): random generator seed

    Returns:
        tuple: tuple of shuffled datsets (np.ndarray)

    """
    if ori_rho is None:
        ori_rho = rho

    n = len(a)
    pearson_r = 2 * np.sin(rho * np.pi / 6)
    rho_mat = np.array([[1, pearson_r], [pearson_r, 1]])
    size = rho_mat.shape[0]
    means = np.zeros(size)
    u = rng.multivariate_normal(means, rho_mat, size=n)
    i1 = np.argsort(u[:, 0])
    i2 = np.argsort(u[:, 1])
    x1, x2 = np.zeros(n), np.zeros(n)
    x1[i1] = a
    x2[i2] = b

    # check if desired rho is achieved
    result, _ = spearmanr(x1, x2)
    # recursive until reaches 2 decimal point accuracy
    if ori_rho - result > 0.01:  # aim for higher rho
        x1, x2 = fit_rho(a, b, rho + 0.01, rng, ori_rho=ori_rho)
    elif ori_rho - result < -0.01:  # aim for lower rho
        x1, x2 = fit_rho(a, b, rho - 0.01, rng, ori_rho=ori_rho)

    return (x1, x2)


