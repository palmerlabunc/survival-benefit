import yaml
import pandas as pd
from lifelines import CoxPHFitter

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