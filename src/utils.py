import yaml
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, wilcoxon, ttest_rel
from typing import List, Tuple

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
                        ax_width: float = 1.5, 
                        ax_height: float = 1.5,
                        max_cols: int = 4) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
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
        ncol = max_cols
    else:
        ncol = n_axes
    if n_axes % max_cols != 0:
        nrow += 1

    if n_axes > 1:
        fig, axes = plt.subplots(nrow, ncol, 
                                 figsize=(ncol*ax_width + 0.2, nrow*ax_height + 0.2),
                                 sharey=True, 
                                 layout='constrained')
    else:
        fig, axes = plt.subplots(figsize=(ax_width, ax_height))
        axes = np.array([axes])
    
    axes = axes.flatten()
    return fig, axes


def get_xticks(tmax: float, metric='months') -> List[float]:
    """Set x-axis limits and ticks.

    Args:
        tmax (float): Maximum time.
        metric (str, optional): Metric (either months or days). Defaults to 'months'.
    Returns:
        List[float]: List of xticks.
    """
    if metric == 'months':
        if tmax < 6:
            return [0, 2, 4]
        elif tmax >= 6 and tmax < 12:
            step = 3
        elif tmax >= 12 and tmax < 24:
            step = 6
        else:
            step = 12
     
    elif metric == 'days':
        if tmax < 100:
            step = 20
        elif tmax >=100 and tmax < 200:
            step = 50
        else:
            step = 100
    
    max_tick = step * int(tmax / step) + 1
    xticks = list(range(0, max_tick, step))
    return xticks


def add_endpoint_to_key(key: str, endpoint: str) -> str:
    """Add endpoint to key.

    Args:
        key (str): Key.
        endpoint (str): Endpoint.

    Returns:
        str: New key.
    """

    tokens = key.split('_')
    # normally key is in the format of {Cancer Type}_{First Author/Year}_{Drug}-{Control}
    # sometimes it will have additional information, 
    # e.g., {Cancer Type}_{First Author/Year}_{Drug}-{Control}_{Additional Info}
    # In this case, we add the endpoint before the additional info 
    # e.g., {Cancer Type}_{First Author/Year}_{Drug}-{Control}_{Endpoint}_{Additional Info}
    
    if len(tokens) > 3:
       return f'{tokens[0]}_{tokens[1]}_{tokens[2]}_{endpoint}_{tokens[3]}'
    return f'{key}_{endpoint}'


def separate_key_endpoint_from_name(combo_endpoint: str) -> Tuple[str, str]:
    """Get key (i.e., {Cancer Type}_{First Author/Year}_{Drug}-{Control}_{Additionl Info}) from
    {Cancer Type}_{First Author/Year}_{Drug}-{Control}_{Endpoint}_{Additional Info}

    Args:
        combo_endpoint (str): Combo name with endpoint.

    Returns:
        str: Key without endpoint
        str: Endpoint
    """
    tokens = combo_endpoint.split('_')
    # 0: cancer  type
    # 1: first author/year
    # 2: drug-control
    # 3: endpoint
    # 4: additional info
    # {Cancer Type}_{First Author/Year}_{Drug}-{Control}_{Endpoint}_{Additionl Info}
    if len(tokens) > 4:
        return '_'.join(tokens[:3]) + '_' + tokens[-1], tokens[3]
    # {Cancer Type}_{First Author/Year}_{Drug}-{Control}_{Endpoint}
    return '_'.join(tokens[:-1]), tokens[-1]


def get_control_name_from_combo(combo: str) -> str:
    """Get control name from combo name.

    Args:
        combo (str): Combo name.

    Returns:
        str: Control name.
    """
    tokens = combo.split('_')
    drugs = tokens[1].split('-')

    return '_'.join([tokens[0], drugs[1]] +  tokens[2:])


def endpoint_simple(endpoint: str) -> str:
    if endpoint == 'OS':
        return 'OS'
    return 'Surrogate'


def test_diff(a: pd.Series,
              b: pd.Series, method=None) -> tuple[str, float, float]:
    """Test difference in Gini coefficients between monotherapy and inferred benefit.
    Use Shapiro test to detirmine normality. If normal, use paired t-test. Otherwise, use Wilcoxon.
    If method (ttest_rel or wilcoxon) is provided, use that method.

    Args:
        a (pd.Series): quantities of interest 1
        b (pd.Series): quantities of interest 2
        method (str, optional): Method to use (ttest_rel or wilcoxon). Defaults to None.

    Returns:
        str: test name
        float: statistics
        float: p-value
    """
    _, shapiro_p = shapiro(a - b)
    print("Shapiro-Wilk test for normality: p = ", shapiro_p)
    if method is None:
        if shapiro_p < 0.05:
            stat, p = wilcoxon(a, b)
            return "Wilcoxon", stat, p
        else:
            stat, p = ttest_rel(a, b)
            return "paired t-test", stat, p
    
    if method == 'ttest_rel':
        stat, p = ttest_rel(a, b)
        return "paired t-test", stat, p
    elif method == 'wilcoxon':
        stat, p = wilcoxon(a, b)
        return "Wilcoxon", stat, p
    else:
        raise ValueError("Invalid method. Choose either ttest_rel or wilcoxon.")