import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import argparse
from scipy.interpolate import interp1d
from scipy.stats import spearmanr


def interpolate(df, x='Time', y='Survival', kind='zero'):
    return interp1d(df[x], df[y], kind=kind, fill_value='extrapolate')


def populate_N_patients(ori_df, N):
    """Scale to make N patients from survival 0 to 100.

    Parameters
    ----------
    ori_df : pandas.DataFrame
        Original survival data.

    Returns
    -------
    pandas.DataFrame
        Survival data with N data points.

    """
    df = ori_df.copy()
    # add starting point (survival=100, time=0)
    df = df.append(pd.DataFrame(
        {'Survival': 100, 'Time': 0}, index=[df.shape[0]]))
    min_survival = df['Survival'].min()
    step = 100 / N

    f = interpolate(df, x='Survival', y='Time')  # survival -> time
    if min_survival <= 0:
        points = np.linspace(0, 100 - step, N)
        new_df = pd.DataFrame({'Time': f(points), 'Survival': points})
    else:
        existing_n = int(np.round((100 - min_survival) / 100 * N, 0))
        existing_points = np.linspace(
            100 * (N - existing_n) / N + step, 100 - step, existing_n)

        # pad patients from [0, min_survival]
        new_df = pd.DataFrame({'Time': df['Time'].max(),
                               'Survival': np.linspace(0, 100 * (N - existing_n) / N, N - existing_n)})
        new_df = new_df.append(pd.DataFrame({'Survival': existing_points,
                                             'Time': f(existing_points)}))
        new_df = new_df.round(
            {'Time': 5, 'Survival': int(np.ceil(-np.log10(step)))})
    assert new_df.shape[0] == N
    return new_df[['Time', 'Survival']].sort_values('Survival').reset_index(drop=True)


def fit_rho3(a, b, rho, rng, ori_rho=None):
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
        x1, x2 = fit_rho3(a, b, rho + 0.01, rng, ori_rho=ori_rho)
    elif ori_rho - result < -0.01:  # aim for lower rho
        x1, x2 = fit_rho3(a, b, rho - 0.01, rng, ori_rho=ori_rho)

    return (x1, x2)


def sample_joint_response_add(ori_a: pd.DataFrame, ori_b: pd.DataFrame, 
                              subtracted: str, scan_time: float) -> list:
    """Calculates predicted PFS time for n-patients in combination therapy under additivity.

    Args:
        ori_a (pd.DataFrame): survival data for treatment A
        ori_b (pd.DataFrame): survival data for treatment B
        subtracted (str): treatment to subtract initial scan time from ('a' or 'b')
        scan_time (float): first scan time

    Returns:
        list: Sorted PFS times
    """
    # ensure that a + b > a and a + b > b
    if subtracted == 'a':
        adjusted = ori_a - scan_time
        adjusted[adjusted < 0] = 0
        predicted = np.maximum(adjusted + ori_b, ori_a)
    elif subtracted == 'b':
        adjusted = ori_b - scan_time
        adjusted[adjusted < 0] = 0
        predicted = np.maximum(ori_a + adjusted, ori_b)
    else:
        print("wrong argument")
    return sorted(predicted, reverse=True)


def sample_joint_response(ori_a: pd.DataFrame, ori_b: pd.DataFrame) -> list:
    """ Calculate predicted PFS time for n-patients in combination therapy under HSA.

    Args:
        ori_a (pd.DataFrame): survival data for treatment A
        ori_b (pd.DataFrame): survival data for treatment B

    Returns:
        list: list of predicted PFS for n-patients in combination therapy under HSA
    """
    return sorted(np.maximum(ori_a, ori_b), reverse=True)


def set_tmax(df_a: pd.DataFrame, df_b: pd.DataFrame, df_ab: pd.DataFrame) -> float:
    """Find minimum of the maximum follow-up time between trials.

    Args:
        df_a (pd.DataFrame): Survival data for drug A
        df_b (pd.DataFrame): Survival data for drug B
        df_ab (pd.DataFrame): Survival data for A+B

    Returns:
        float: max time
    """    
    if df_a.at[0, 'Survival'] < 5 and df_b.at[0, 'Survival'] < 5:
        tmax = min(
            max(df_a.at[0, 'Time'], df_b.at[0, 'Time']), df_ab.at[0, 'Time'])
    else:
        tmax = min(df_a['Time'].max(), df_b['Time'].max())
    return tmax


def predict_both(df_a: pd.DataFrame, df_b: pd.DataFrame, 
                 name_a: str, name_b: str, subtracted: str, scan_time: float, 
                 df_ab=None, N=5000, rho=0.3, seed_ind=0, seed_add=0, save=True, outdir=None) -> tuple:
    """ Predict combination effect using HSA and additivity model and writes csv output.

    Args:
        df_a (pd.DataFrame): survival data for treatment A (Experimental)
        df_b (pd.DataFrame): survival data for treatment B (Control)
        name_a (str): treatment A name
        name_b (str): treatment B name
        subtracted (str): treatment arm to substract first scan time from ('a' or 'b')
        scan_time (float): first scan time
        df_ab (pd.DataFrame, optional): survival data for treatment A+B. Defaults to None.
        N (int, optional): number of virtual patients. Defaults to 5000.
        rho (float, optional): correlation value. Defaults to 0.3.
        seed_ind (int): random generator seed for independent model. Defaults to 0.
        seed_add (int): random generator seed for additivity model. Defaults to 0.
        save (bool): export data to csv. Defaults to True.
        outdir (str): directory to save exported data. If None, save in current directory. Defaults to None. 
    
    Returns:
        pd.DataFrame : HSA prediction
        pd.DataFrame : additivity prediction
    """
    a = populate_N_patients(df_a, N)
    b = populate_N_patients(df_b, N)
    
    patients = a['Survival'].values
    rng_ind = np.random.default_rng(seed_ind)
    new_ind_a, new_ind_b = fit_rho3(a['Time'].values, b['Time'].values, rho, rng_ind)
    independent = pd.DataFrame({'Time': sample_joint_response(new_ind_a, new_ind_b), 
                                'Survival': patients})
    
    rng_add = np.random.default_rng(seed_add)
    new_add_a, new_add_b = fit_rho3(a['Time'].values, b['Time'].values, rho, rng_add)
    additivity = pd.DataFrame({'Time': sample_joint_response_add(new_add_a, new_add_b, subtracted, scan_time),
                               'Survival': patients})

    additivity = additivity.sort_values('Survival', ascending=True).reset_index(drop=True)
    independent = independent.sort_values('Survival', ascending=True).reset_index(drop=True)
    
    if df_ab is not None:
        tmax = set_tmax(df_a, df_b, df_ab)
    else:
        tmax = min(df_a['Time'].max(), df_b['Time'].max())
    
    independent.loc[independent['Time'] > tmax, 'Time'] = tmax
    additivity.loc[additivity['Time'] > tmax, 'Time'] = tmax

    if save == True:
        if outdir is not None:
            additivity.round(5).to_csv(f'{outdir}/{name_a}-{name_b}_combination_predicted_add.csv', 
                                    index=False)
            independent.round(5).to_csv(f'{outdir}/{name_a}-{name_b}_combination_predicted_ind.csv',
                                        index=False)
        else:
            additivity.round(5).to_csv(f'{name_a}-{name_b}_combination_predicted_add.csv', 
                                    index=False)
            independent.round(5).to_csv(f'{name_a}-{name_b}_combination_predicted_ind.csv',
                                        index=False)
    
    return (independent, additivity)


def subtract_which_scan_time(scan_a: float, scan_b: float) -> tuple:
    """Determines which monotherapy scan time to subtract.

    Args:
        scan_a (float): Scan time of drug A (Experimental)
        scan_b (float): Scan time of drug B (Control)

    Returns:
        tuple: ('a' or 'b', scan_time)
    """    
    if scan_a == 9999:
        scan_a = -9999
    if scan_b == 9999:
        scan_b = -9999
    if scan_a < scan_b:
        scan_time = scan_b
        subtracted = 'b'
    else:
        scan_time = scan_a
        subtracted = 'a'
    return (subtracted, scan_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, 
                        help='Dataset to use (approved, all_phase3, placebo')
    args = parser.parse_args()
    
    config_dict = CONFIG[args.dataset]
    sheet = config_dict['metadata_sheet_seed']
    data_dir = config_dict['data_dir']
    pred_dir = config_dict['pred_dir']

    indf = pd.read_csv(sheet, sep='\t')
    for i in indf.index:
        name_a = indf.at[i, 'Experimental']
        name_b = indf.at[i, 'Control']
        name_ab = indf.at[i, 'Combination']
        corr = indf.at[i, 'Corr']  # experimental spearman correlation value
        # random generator seed that results in median of 100 simulations
        seed_ind = indf.at[i, 'ind_median_run']
        seed_add = indf.at[i, 'add_median_run']
        df_a = pd.read_csv(f'{data_dir}/{name_a}.clean.csv',
                           header=0, index_col=False)
        df_b = pd.read_csv(f'{data_dir}/{name_b}.clean.csv',
                           header=0, index_col=False)
        df_ab = pd.read_csv(f'{data_dir}/{name_ab}.clean.csv',
                            header=0, index_col=False)
        # subtract initial scan time of the larger one
        scan_a = indf.at[i, 'Experimental First Scan Time']
        scan_b = indf.at[i, 'Control First Scan Time']

        subtracted, scan_time = subtract_which_scan_time(scan_a, scan_b)

        predict_both(df_a, df_b, name_a, name_b, subtracted, scan_time,
                     df_ab=df_ab, rho=corr, seed_ind=seed_ind, seed_add=seed_add, outdir=pred_dir)


if __name__ == "__main__":
    main()
