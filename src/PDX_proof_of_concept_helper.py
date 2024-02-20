import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from lifelines import KaplanMeierFitter
from typing import Collection, Tuple, Mapping
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_cox_results
from survival_benefit.survival_benefit_class import SurvivalBenefit
from survival_benefit.utils import interpolate
from utils import load_config
from PDX_proof_of_concept_helper import *

config = load_config()['PDX']
DELTA_T = config['delta_t']
MIN_N = 30

def get_models(dat: pd.DataFrame, tumor: str, drug1: str, drug2: str = None, combo=True) -> np.array:
    """Returns the list of PDX models that are treated with drug1, drug2 (if specified), 
    and drug1 + drug2 (if combo is True). The model must be treated with all specified drugs
    and combination.

    Args:
        dat (pd.DataFrame): _description_
        tumor (str): _description_
        drug1 (str): _description_
        drug2 (str, optional): _description_. Defaults to None.
        combo (bool, optional): _description_. Defaults to True.

    Returns:
        np.array: array of PDX models
    """
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


def get_event_table(dat: pd.DataFrame, models: Collection, drug: str, 
                    strict_censoring=False, tmax=100) -> pd.DataFrame:
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
    try:
        combo_list.remove(['BYL719', 'cetuximab', 'encorafenib',
                        'BYL719 + cetuximab + encorafenib'])
    except ValueError:
        pass

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


def make_event_dataframe_dict_mono(dat: pd.DataFrame, logrank: pd.DataFrame) -> Mapping[Tuple[str, str], pd.DataFrame]:
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


def make_event_dataframe_dict_combo(dat: pd.DataFrame, logrank: pd.DataFrame, strict_censoring=False) -> Mapping[Tuple[str, str], pd.DataFrame]:
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


def compute_benefits_helper(event_df: pd.DataFrame, 
                            drug1_name: str, drug2_name: str, 
                            control_drug_idx: int, 
                            r: float,
                            tmax: float = 100) -> Tuple[float, pd.DataFrame]:
    
    assert control_drug_idx in [1, 2], "control_drug_idx must be 1 or 2"
    km_comb = get_km_curve(event_df, 'T_comb', 'E_comb')
    
    km_mono = get_km_curve(event_df, 
                           f'T_mono{control_drug_idx}', 
                           f'E_mono{control_drug_idx}')
    
    if control_drug_idx == 1:
        ctrl_drug_name = drug1_name
        combo_name = drug2_name + '-' + drug1_name
    
    else:
        ctrl_drug_name = drug2_name
        combo_name = drug1_name + '-' + drug2_name
    
    sb = SurvivalBenefit(km_mono, km_comb,
                         ctrl_drug_name, combo_name,
                         save_mode=False)

    sb.set_tmax(tmax)

    # use correlation from bestAvgResponse
    sb.compute_benefit_at_corr(r, use_bestmatch=True)
    actual_corr = sb.corr_rho_actual
    actual_corr_benefit = sb.benefit_df.copy()

    return (actual_corr, actual_corr_benefit)


def compute_rmse_in_survival(true_benefit: pd.Series, inferred_benefit: pd.DataFrame) -> np.array:
    delta_t = inferred_benefit[DELTA_T].sort_values(ascending=True).values
    inferred_survival = np.linspace(0, 100, delta_t.size)
    inferred = pd.DataFrame({'survival': inferred_survival, DELTA_T: delta_t})
    f_inferred = interpolate(inferred, x=DELTA_T, y='survival')
    
    true_delta_t = true_benefit.sort_values(ascending=True)
    true_survival = np.linspace(0, 100, true_delta_t.size)
    true = pd.DataFrame({'survival': true_survival, DELTA_T: true_delta_t})
    f_true = interpolate(true, x=DELTA_T, y='survival')

    t = np.linspace(0, 100, 100)
    
    return np.sqrt(np.mean(np.square(f_true(t) - f_inferred(t))))


def compute_rmse_in_time(true_benefit: pd.Series, inferred_benefit: pd.DataFrame) -> np.array:
    delta_t = inferred_benefit[DELTA_T].sort_values(ascending=False).values
    inferred_survival = np.linspace(0, 100, delta_t.size)
    inferred = pd.DataFrame({'Survival': inferred_survival, 'Time': delta_t})
    f = interpolate(inferred, x='Survival', y='Time')
    
    true_benefit[true_benefit < 0] = 0  # disregard negative values
    true_benefit = np.sort(true_benefit)[::-1] # descending
    surv_idx = np.linspace(0, 100 - 100 / true_benefit.size, true_benefit.size)
    
    return np.sqrt(np.mean(np.square(f(surv_idx) - true_benefit)))


def compute_rmse_in_survival_km(true_benefit_event_df: pd.DataFrame, inferred_deltat: pd.Series, 
                                tmax: float = None) -> float:
    inferred_deltat.loc[inferred_deltat < 0] = 0  # disregard negative values

    true_benefit_km = KaplanMeierFitter()
    true_benefit_km.fit(true_benefit_event_df['T_benefit'], 
                        true_benefit_event_df['E_benefit'])
    
    inferred_benefit_km = KaplanMeierFitter()
    inferred_benefit_km.fit(inferred_deltat)

    if tmax is None:
        tmax = true_benefit_event_df['T_benefit'].max()
    
    ts = np.linspace(0, tmax, 1000)

    # convert to 0 - 100 % scale
    true_benefit_surv = 100 * true_benefit_km.survival_function_at_times(ts)
    inferre_benefit_surv = 100 * inferred_benefit_km.survival_function_at_times(ts)

    return np.sqrt(np.mean((true_benefit_surv - inferre_benefit_surv)**2))


def added_benefit_event_table(event_df: pd.DataFrame, control_drug_idx: int) -> pd.DataFrame:
    """Only uses models where the control drug event is observed.

    Args:
        event_df (pd.DataFrame): IPD for combo, mono1, and mono2
        control_drug_idx (int): _description_

    Returns:
        pd.DataFrame: individual patient data for added benefit
    """
    df = event_df.copy()
    # only use models where the control drug event is observed
    df = df[df[f'E_mono{control_drug_idx}'] == 1]
    df.loc[:, 'T_mono'] = df[f'T_mono{control_drug_idx}']
    df.loc[:, 'T_benefit'] = df['T_comb'] - df[f'T_mono{control_drug_idx}']
    # if the benefit is negative, set it to 0
    df.loc[df['T_benefit'] < 0, 'T_benefit'] = 0
    df.loc[:, 'E_benefit'] = df['E_comb']

    return df[['T_benefit', 'E_benefit']]

##### OBSOLETE FUNCTIONS (NO LONGER USED) #####

def plot_correlation_benefit_comparison_3lineplot(result_df: pd.DataFrame) -> plt.Figure:
    melted = pd.melt(result_df,
                     id_vars=['Tumor Type', 'Combination', 'Effect Drug'],
                     value_vars=['RMSE_r_BestAvgRes', 'RMSE_r_A_deltaB', 'RMSE_r_Highest'])

    melted.loc[:, 'id'] = melted['Tumor Type'] + ':|:' + \
        melted['Combination'] + ':|:' + melted['Effect Drug']

    melted.loc[:, 'variable'] = pd.Categorical(melted['variable'],
                                               categories=['RMSE_r_A_deltaB', 'RMSE_r_BestAvgRes', 'RMSE_r_Highest'],
                                               ordered=True)

    fig, ax = plt.subplots(figsize=(3, 4))
    sns.pointplot(melted, x='variable', y='value', hue='id',
                  ax=ax)
    ax.set_xticklabels(['A_deltaB', 'BestAvgResponse', 'High'])
    ax.set_xlabel('Correlation')
    ax.set_ylabel('RMSE (days)')
    ax.legend().remove()
    

def plot_correlation_benefit_comparison_3boxplot(result_df: pd.DataFrame) -> plt.Figure:
    melted = pd.melt(result_df,
                     id_vars=['Tumor Type', 'Combination', 'Effect Drug'],
                     value_vars=['RMSE_r_A_deltaB', 'RMSE_r_BestAvgRes', 'RMSE_r_Highest'])

    melted.loc[:, 'id'] = melted['Tumor Type'] + ':|:' + \
        melted['Combination'] + ':|:' + melted['Effect Drug']

    melted.loc[:, 'variable'] = pd.Categorical(melted['variable'],
                                            categories=['RMSE_r_A_deltaB', 'RMSE_r_BestAvgRes', 'RMSE_r_Highest'],
                                            ordered=True)
    
    fig, ax = plt.subplots(figsize=(3, 4))
    sns.boxplot(melted, x='variable', y='value',
                ax=ax)
    sns.stripplot(melted, x='variable', y='value',
                  ax=ax)
    ax.set_xticklabels(['A_deltaB', 'BestAvgResponse', 'High'])
    ax.set_xlabel('Correlation')
    ax.set_ylabel('RMSE (days)')

    return fig


def plot_correlation_benefit_comparison_2boxplot(result_df: pd.DataFrame) -> plt.Figure:
    melted = pd.melt(result_df,
                     id_vars=['Tumor Type', 'Combination', 'Effect Drug'],
                     value_vars=['RMSE_r_BestAvgRes', 'RMSE_r_Highest'])

    melted.loc[:, 'id'] = melted['Tumor Type'] + ':|:' + \
        melted['Combination'] + ':|:' + melted['Effect Drug']
    
    stat, p = wilcoxon(result_df['RMSE_r_Highest'], 
                       result_df['RMSE_r_BestAvgRes'])

    fig, ax = plt.subplots(figsize=(3, 4))
    sns.boxplot(melted, x='variable', y='value',
                ax=ax)
    sns.stripplot(melted, x='variable', y='value',
                  ax=ax)
    ax.set_xticklabels(['BestAvgResponse', 'High'])
    ax.set_xlabel('Correlation')
    ax.set_ylabel('RMSE (days)')
    ax.set_title(f'two-sided Wilcoxon test (p={p:.2f})')
    return fig
