import yaml
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import numpy as np

def load_config():
    file_path = 'config.yaml'
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def get_cox_results(ipd_base: pd.DataFrame, ipd_test: pd.DataFrame) -> tuple:
    """Perform Cox PH test. IPD should have columns Time, Event.
    HR < 1 indicates that test has less hazard (i.e., better than) base.

    Args:
        ipd_base (pd.DataFrame): IPD of control arm.
        ipd_test (pd.DataFrame): IPD of test arm. 

    Returns:
        (float, float, float, float): p, HR, lower 95% CI, upper 95% CI
    """    
    cph = CoxPHFitter()
    ipd_base.loc[:, 'Arm'] = 0
    ipd_test.loc[:, 'Arm'] = 1
    merged = pd.concat([ipd_base, ipd_test],
                        axis=0).reset_index(drop=True)
    cph.fit(merged, duration_col='Time', event_col='Event')
    return tuple(cph.summary.loc['Arm', ['p', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']])


def set_figure_size_dim(n_axes: int = 1, 
                        ax_width: float = 1.5, ax_height: float = 1.5,
                        max_cols: int = 4) -> (plt.Figure, np.ndarray[plt.Axes]):
    """Generate a figure with size based on number of columns and desired axis size.

    Args:
        n_axes (int, optional): Number of axes. Defaults to 1.
        ax_width (float, optional): Width of each axis. Defaults to 1.5.
        ax_height (float, optional): Height of each axis. Defaults to 1.5.
        max_cols (int, optional): Maximum number of columns. Defaults to 4.
    
    Returns:
        (plt.Figure, np.array[plt.Axes]): Figure and axes.
    """
    nrow = int(n_axes/max_cols)
    if nrow > 0:
        ncol = 5
    else:
        ncol = n_axes
    if n_axes % max_cols != 0:
        nrow += 1

    if n_axes > 1:
        fig, axes = plt.subplots(nrow, ncol, 
                                 figsize=(ncol*ax_width + 0.2, nrow*ax_height + 0.2), 
                                 layout='constrained')
    else:
        fig, axes = plt.subplots(figsize=(ax_width, ax_height))
        axes = np.array([axes])
    
    axes = axes.flatten()
    return fig, axes


