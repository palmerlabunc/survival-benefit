import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from typing import Collection
from utils import get_cox_results

MIN_N = 30

def get_models(dat: pd.DataFrame, tumor: str, drug1: str, drug2: str = None, combo=True) -> np.array:
    if drug2 is None:
        return dat[(dat['Treatment'] == drug1) & (dat['Tumor Type'] == tumor)]['Model'].unique()

    models1 = dat[(dat['Treatment'] == drug1) & (
        dat['Tumor Type'] == tumor)]['Model'].unique()
    models2 = dat[(dat['Treatment'] == drug2) & (
        dat['Tumor Type'] == tumor)]['Model'].unique()
    
    if combo:
        assert drug2 is not None, "drug2 must be specified"
        models3 = dat[(dat['Treatment'] == drug1 + ' + ' + drug2)
                      & (dat['Tumor Type'] == tumor)]['Model'].unique()
        return np.intersect1d(np.intersect1d(models1, models2), models3)
    return np.intersect1d(models1, models2)


def get_event_table(dat: pd.DataFrame, models: Collection, drug: str, strict_censoring=False, tmax=100):
    """ Get event table allow censoring (end of follow-up OR 
    if TimeToDouble == Day_Last and ResponseCategory is not PD). 
    If stict_censoring is True, mark as censored
    only when TimeToDouble > tmax. Otherwise, mark as 
    censored when TimeToDouble == Day_Last and ResponseCategory is not PD.
    """
    if models is None:
        tmp = dat[dat['Treatment'] == drug].sort_values('Model')
    else:
        tmp = dat[(dat['Model'].isin(models)) & (
            dat['Treatment'] == drug)].sort_values('Model')
    # only allow censoring due to end of follow-up
    if strict_censoring:
        event = ~(tmp['TimeToDouble'] > tmax)
    # if TimeToDouble == Day_Last and ResponseCategory is not PD, mark as censored
    else:
        event=~((tmp['TimeToDouble'] == tmp['Day_Last'])
                & (tmp['ResponseCategory'] != 'PD'))
        
    tmp.loc[:, 'Event'] = event.astype(int)
    tmp.loc[:, "Time"] = tmp['TimeToDouble'].astype(float)
    tmp = tmp.set_index('Model')
    tmp = tmp[["Time", 'Event']]
    return tmp


def get_number_of_models_for_all_combo(dat: pd.DataFrame) -> pd.DataFrame:
    combo_arr = dat[dat['Treatment type'] == 'combo']['Treatment'].unique()
    combo_list = [s.split(' + ') + [s] for s in combo_arr]
    # remove three-drug combination
    combo_list.remove(['BYL719', 'cetuximab', 'encorafenib',
                    'BYL719 + cetuximab + encorafenib'])

    tmp = []

    for drug1, drug2, comb in combo_list:
        for tumor in dat[(dat['Treatment'] == comb)]['Tumor Type'].unique():
            models = get_models(dat, tumor, drug1, drug2, combo=True)
            tmp.append([tumor, comb, len(models)])
    
    ndf = pd.DataFrame(tmp, columns=['Tumor Type', 'Combination', 'N'])
    return ndf


def get_km_curve(event_df: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
    """Get km curve from event dataframe.

    Args:
        event_df (pd.DataFrame): event dataframe
        time_col (str): time column name
        event_col (str): event column name

    Returns:
        pd.DataFrame: km curve
    """
    kmf = KaplanMeierFitter()
    kmf.fit(event_df[time_col], event_df[event_col])
    km_df = kmf.survival_function_.reset_index()
    km_df.columns = ['Time', 'Survival']
    km_df.loc[:, 'Survival'] = 100 * km_df['Survival']
    return km_df


def make_event_dataframe_dict_mono(dat: pd.DataFrame, logrank: pd.DataFrame) -> dict[(str, str), pd.DataFrame]:
    """Make event dataframe dictionary with keys (tumor, drug) for monotherapy vs. untreated.

    Args:
        dat (pd.DataFrame): PDX longform dataframe
        logrank (pd.DataFrame): logrank or coxph dataframe

    Returns:
        dict[(str, str), pd.DataFrame]: keys are (tumor, drug) tuples; values are event dataframes
    """
    df_dict = {}
    for i in logrank.index:
        tumor = logrank.at[i, 'Tumor Type']
        drug = logrank.at[i, 'Treatment']
        models = get_models(dat, tumor, 'untreated', drug, combo=False)
        mono = get_event_table(dat, models, drug)
        unt = get_event_table(dat, models, 'untreated')

        df = pd.concat([mono, unt], axis=1, join='outer')
        df.columns = ['T_mono', 'E_mono', 'T_untreated', 'E_untreated']
        df_dict[(tumor, drug)] = df
    return df_dict


def make_event_dataframe_dict_combo(dat: pd.DataFrame, logrank: pd.DataFrame, strict_censoring=False) -> dict[(str, str), pd.DataFrame]:
    """Make event dataframe dictionary with keys (tumor, combination) for combination vs. monotherapies (both).

    Args:
        dat (pd.DataFrame): PDX longform dataframe
        logrank (pd.DataFrame): logrank or coxph dataframe
        strict_censoring (bool, optional): If True, censoring is only due to end of follow-up. Defaults to False.

    Returns:
        dict[(str, str), pd.DataFrame]: keys are (tumor, combination) tuples; values are event dataframes
    """
    df_dict = {}
    for i in logrank.index:
        tumor = logrank.at[i, 'Tumor Type']
        drug_comb = logrank.at[i, 'Combination']
        tokens = drug_comb.split(' + ')
        drug1 = tokens[0]
        drug2 = tokens[1]
        models = get_models(dat, tumor, drug1, drug2)

        comb = get_event_table(dat, models, drug_comb, 
                               strict_censoring=strict_censoring)
        mono1 = get_event_table(dat, models, drug1, 
                                strict_censoring=strict_censoring)
        mono2 = get_event_table(dat, models, drug2, 
                                strict_censoring=strict_censoring)
        unt = get_event_table(dat, models, 'untreated', 
                              strict_censoring=strict_censoring)
    
        df = pd.concat([comb, mono1, mono2, unt], axis=1, join='outer')
        df.columns = ['T_comb', 'E_comb', 'T_mono1', 'E_mono1',
                      'T_mono2', 'E_mono2', 'T_untreated', 'E_untreated']
        df_dict[(tumor, drug_comb)] = df
    return df_dict


def coxph_monotherapy_vs_untreated(dat: pd.DataFrame) -> pd.DataFrame:
    mono_arr = dat[dat['Treatment type'] == 'single']['Treatment'].unique()
    # as tuple of (tumor type, drug)
    mono_arr = dat[dat['Treatment'].isin(
        mono_arr)][['Tumor Type', 'Treatment']].dropna()
    mono_arr = mono_arr.drop_duplicates().to_numpy()

    # cox ph test between untreated vs. monotherapy
    arr = []
    for tumor, drug in mono_arr:
        if drug == 'untreated':
            continue

        models = get_models(dat, tumor, drug)
        if models.shape[0] < MIN_N:
            continue

        mono = get_event_table(dat, models, drug)
        unt = get_event_table(dat, models, 'untreated')

        p, hr, _, _ = get_cox_results(unt, mono)
        arr.append([tumor, drug, mono.shape[0], unt.shape[0], hr, p])

    coxph = pd.DataFrame(arr,
                         columns=['Tumor Type', 'Treatment', 'n_treat', 'n_untreat', 'HR', 'p'])
    return coxph


def coxph_combo_vs_mono(dat: pd.DataFrame, strict_censoring=False) -> pd.DataFrame:
    ndf = get_number_of_models_for_all_combo(dat)
    ndf = ndf[ndf['N'] >= MIN_N]
    lr_tmp = []

    for i in ndf.index:
        tumor = ndf.at[i, 'Tumor Type']
        drug_comb = ndf.at[i, 'Combination']
        tokens = drug_comb.split(' + ')
        drug1 = tokens[0]
        drug2 = tokens[1]
        models = get_models(dat, tumor, drug1, drug2)
        comb = get_event_table(dat, models, drug_comb,
                               strict_censoring=strict_censoring)
        mono1 = get_event_table(
            dat, models, drug1, strict_censoring=strict_censoring)
        mono2 = get_event_table(
            dat, models, drug2, strict_censoring=strict_censoring)

        p1, hr1, _, _ = get_cox_results(mono1, comb)
        p2, hr2, _, _ = get_cox_results(mono2, comb)

        # save result
        lr_tmp.append([tumor, drug_comb, len(models), hr1, hr2, p1, p2])

    coxph_combo = pd.DataFrame(lr_tmp,
                               columns=['Tumor Type', 'Combination', 'N', 'HR1', 'HR2', 'p1', 'p2'])

    return coxph_combo

