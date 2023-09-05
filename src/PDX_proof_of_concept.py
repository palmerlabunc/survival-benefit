from pathlib import Path
from multiprocessing import Pool
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon
from lifelines import KaplanMeierFitter, WeibullFitter, CoxPHFitter
import warnings
from typing import Collection
from survival_benefit.survival_benefit_class import SurvivalBenefit
from survival_benefit.utils import interpolate
from PDXE_correlation import get_pdx_corr_data
from utils import load_config

# Three main analyses:
# 1. Is there anatognism in combination threapies?
# 2. How much difference is there betweeen corr(A, deltaB) and corr(A, B)?
# 3. Does using corr(A, B) better infer the benefit profile than using high correlation?

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
        
    tmp.loc[:, 'E'] = event.astype(int)
    tmp.loc[:, 'T'] = tmp['TimeToDouble'].astype(float)
    tmp = tmp.set_index('Model')
    tmp = tmp[['T', 'E']]
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


def get_cox_results(ipd_base: pd.DataFrame, ipd_test: pd.DataFrame) -> tuple:
    """Perform Cox PH test. IPD should have columns T (Time), E (Event).
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
    cph.fit(merged, duration_col='T', event_col='E')
    return tuple(cph.summary.loc['Arm', ['p', 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%']])


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

        comb = get_event_table(dat, models, drug_comb, strict_censoring=strict_censoring)
        mono1 = get_event_table(dat, models, drug1, strict_censoring=strict_censoring)
        mono2 = get_event_table(dat, models, drug2, strict_censoring=strict_censoring)
        unt = get_event_table(dat, models, 'untreated', strict_censoring=strict_censoring)
    
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

    # logrank test between untreated vs. monotherapy
    arr = []
    for tumor, drug in mono_arr:
        if drug == 'untreated':
            continue

        models = get_models(dat, tumor, drug)
        if models.shape[0] < 20:
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
    ndf = ndf[ndf['N'] >= 20]
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


def prepare_dataframe_for_stripplot(dat: pd.DataFrame) -> pd.DataFrame:
    coxph_combo = coxph_combo_vs_mono(dat)
    event_df_combo_dict = make_event_dataframe_dict_combo(dat, coxph_combo)

    merged = pd.DataFrame(columns=['Tumor Type', 'Combination',
                                   'Model', 'target', 'added_benefit', 'event_observed'])

    for i in range(coxph_combo.shape[0]):
        p1 = coxph_combo.at[i, 'p1']
        p2 = coxph_combo.at[i, 'p2']
        hr1 = coxph_combo.at[i, 'HR1']
        hr2 = coxph_combo.at[i, 'HR2']
        if not (p1 < 0.05 or p2 < 0.05):
            continue
        tumor = coxph_combo.at[i, 'Tumor Type']
        drug_comb = coxph_combo.at[i, 'Combination']
        tokens = drug_comb.split(' + ')
        drug1 = tokens[0]
        drug2 = tokens[1]

        df = event_df_combo_dict[(tumor, drug_comb)]

        if p1 < 0.05 and hr1 < 1:  # find effect of drug2
            print(tumor, drug_comb, drug2)
            df = df[df['E_comb'] + df['E_mono1'] > 0]  # exclude both censored
            # if event only observed in comb AND comb < mono => definately comb < mono
            # if event only observed in mono AND comb > mono => definately comb > mono
            df = df[(df['T_comb'] - df['T_mono1']) *
                    (df['E_comb'] - df['E_mono1']) <= 0]
            tmp = pd.concat([df['T_comb'] - df['T_mono1'],
                            (df['E_comb'] + df['E_mono1'] == 2).astype(int)], axis=1)
            tmp = tmp.reset_index()
            target = drug2

            tmp.columns = ['Model', 'added_benefit', 'event_observed']
            tmp.loc[:, 'target'] = target
            tmp.loc[:, 'Tumor Type'] = tumor
            tmp.loc[:, 'Combination'] = drug_comb
            merged = pd.concat([merged, tmp], axis=0, ignore_index=True)

        if p2 < 0.05 and hr2 < 1:  # find effect of drug1
            print(tumor, drug_comb, drug1)
            df = df[df['E_comb'] + df['E_mono2'] > 0]  # exclude both censored
            # if event only observed in comb AND comb < mono => definately comb < mono
            # if event only observed in mono AND comb > mono => definately comb > mono
            df = df[(df['T_comb'] - df['T_mono2']) *
                    (df['E_comb'] - df['E_mono2']) <= 0]
            tmp = pd.concat([df['T_comb'] - df['T_mono2'],
                            (df['E_comb'] + df['E_mono2'] == 2).astype(int)], axis=1)
            tmp = tmp.reset_index()
            target = drug1

            tmp.columns = ['Model', 'added_benefit', 'event_observed']
            tmp.loc[:, 'target'] = target
            tmp.loc[:, 'Tumor Type'] = tumor
            tmp.loc[:, 'Combination'] = drug_comb
            merged = pd.concat([merged, tmp], axis=0, ignore_index=True)
    
    return merged


def stripplot_added_benefit(data: pd.DataFrame) -> plt.figure:
    plt.style.use('env/publication.mplstyle')
    data.loc[:, 'label'] = 'Effect of ' + \
        data['target'] + ' in ' + data['Combination']
    g = sns.catplot(y="label", x="added_benefit", hue="event_observed",
                    data=data, kind="strip", height=4, aspect=1.5)
    g.fig.axes[0].axvline(0, linestyle='--', color='k')
    plt.xlabel('Added benefit (days)')
    plt.ylabel('')
    return g.fig


def prepare_dataframes_for_distplot(dat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combo_df = prepare_dataframe_for_stripplot(dat)

    coxph_mono = coxph_monotherapy_vs_untreated(dat)
    event_df_dict = make_event_dataframe_dict_mono(dat, coxph_mono)
    
    cols = ['Tumor Type', 'Treatment', 'Model',
            'added_benefit', 'event_observed']

    mono_merged = pd.DataFrame(columns=cols)
    active = pd.DataFrame(columns=cols)
    inactive = pd.DataFrame(columns=cols)

    for i in range(coxph_mono.shape[0]):
        tumor = coxph_mono.at[i, 'Tumor Type']
        drug = coxph_mono.at[i, 'Treatment']

        df = event_df_dict[(tumor, drug)]
        df = df[df['E_mono'] + df['E_untreated'] > 0]  # exclude both censored
        # if event only observed in comb AND comb < mono => definately comb < mono
        # if event only observed in mono AND comb > mono => definately comb > mono
        df = df[(df['T_mono'] - df['T_untreated']) *
                (df['E_mono'] - df['E_untreated']) <= 0]
        tmp = pd.concat([df['T_mono'] - df['T_untreated'],
                        (df['E_mono'] + df['E_untreated'] == 2).astype(int)],
                        axis=1)
        tmp = tmp.reset_index()
        tmp.columns = ['Model', 'added_benefit', 'event_observed']
        tmp.loc[:, 'Tumor Type'] = tumor
        tmp.loc[:, 'Treatment'] = drug

        if coxph_mono.at[i, 'p'] < 0.05 and coxph_mono.at[i, 'HR'] < 1:
            active = pd.concat([active, tmp], axis=0, ignore_index=True)
        else:
            inactive = pd.concat([inactive, tmp], axis=0, ignore_index=True)

        mono_merged = pd.concat([mono_merged, tmp], axis=0, ignore_index=True)

    return mono_merged, active, inactive, combo_df


def distplot_monotherapy_and_combo_added_benefit(mono_merged: pd.DataFrame, mono_active: pd.DataFrame, 
                                                 mono_inactive: pd.DataFrame, combo_active: pd.DataFrame) -> plt.figure:
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(4, 5))
    sns.despine()

    sns.histplot(x='added_benefit', data=mono_merged,
                color=sns.color_palette()[0], kde=True, ax=axes[0])
    sns.histplot(x='added_benefit', data=mono_active,
                color=sns.color_palette()[1], kde=True, ax=axes[1])
    sns.histplot(x='added_benefit', data=mono_inactive,
                color=sns.color_palette()[2], kde=True, ax=axes[2])
    sns.histplot(x='added_benefit', data=combo_active,
                color=sns.color_palette()[3], kde=True, ax=axes[3])
    axes[0].set_title(f'All monotherapy (n={mono_merged.shape[0]})')
    axes[1].set_title(f'Active monotherapy (n={mono_active.shape[0]})')
    axes[2].set_title(f'Inactive monotherapy (n={mono_inactive.shape[0]})')
    axes[3].set_title(f'Successful combination (n={combo_active.shape[0]})')
    axes[3].set_xlabel('Added benefit (day)')
    
    return fig


def cumulative_distplot_monotherapy_and_combo_added_benefit(mono_merged: pd.DataFrame, mono_active: pd.DataFrame,
                                                            mono_inactive: pd.DataFrame, combo_active: pd.DataFrame) -> plt.figure:
    plt.style.use('env/publication.mplstyle')
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(4, 3))
    sns.despine()

    sns.ecdfplot(x='added_benefit', data=mono_merged,
                color=sns.color_palette()[0], ax=ax)
    sns.ecdfplot(x='added_benefit', data=mono_active,
                color=sns.color_palette()[1], ax=ax)
    sns.ecdfplot(x='added_benefit', data=mono_inactive,
                color=sns.color_palette()[2], ax=ax)
    sns.ecdfplot(x='added_benefit', data=combo_active,
                color=sns.color_palette()[3], ax=ax)
    ax.set_xlabel('Added benefit (day)')
    ax.axvline(0, linestyle='--', color='k', alpha=0.5)
    ax.legend(labels=['All monotherapy', 'Active monotherapy',
            'Inactive monotherapy', 'Successful combination'])
    return fig


def merge_dataframes(mono_merged: pd.DataFrame, mono_active: pd.DataFrame,
                     mono_inactive: pd.DataFrame, combo_active: pd.DataFrame) -> pd.DataFrame:
    combo_active.loc[:, 'Treatment'] = combo_active['Combination']
    mono_active.loc[:, 'Activity'] = 'active'
    mono_inactive.loc[:, 'Activity'] = 'inactive'
    merged = pd.concat([mono_active, mono_inactive, combo_active], axis=0,
                       ignore_index=True, join='outer')
    return merged


def corr_AB_vs_corr_A_deltaB(dat: pd.DataFrame) -> pd.DataFrame:
    coxph_df_strict = coxph_combo_vs_mono(dat, strict_censoring=True)
    event_df_dict_strict = make_event_dataframe_dict_combo(dat, coxph_df_strict, strict_censoring=True)
    info = dat[['Model', 'Tumor Type', 'BRAF_mut', 'RAS_mut']
               ].drop_duplicates().set_index('Model')
    tmax = 100
    r_list = []
    for i in coxph_df_strict.index:
        tumor = coxph_df_strict.at[i, 'Tumor Type']
        combo = coxph_df_strict.at[i, 'Combination']
        drug1 = combo.split(' + ')[0]
        drug2 = combo.split(' + ')[1]

        event_df = event_df_dict_strict[(tumor, combo)].copy()
        corr_dat = get_pdx_corr_data(dat, info, drug1, drug2, 
                                     metric='BestAvgResponse')
        # correlation using BestAvgResponse
        r_bestavgres, _ = spearmanr(corr_dat[drug1], corr_dat[drug2])

        # correlation using T_A vs. T_B
        # If censored, use max follow-up time instead
        event_df.loc[event_df['E_mono1'] == 0, 'T_mono1'] = tmax
        event_df.loc[event_df['E_mono2'] == 0, 'T_mono2'] = tmax
        r_t12, _ = spearmanr(event_df['T_mono1'], event_df['T_mono2'])

        # correlation using T_A vs. delta_B (or vice-versa)
        # use only models with event observed in both arms
        event_observed = event_df['E_mono1'] + event_df['E_mono2'] == 2
        event_df['added_benefit_of_drug1'] = event_df['T_comb'] - \
            event_df['T_mono2']
        event_df['added_benefit_of_drug2'] = event_df['T_comb'] - \
            event_df['T_mono1']

        r_1_delta2, _ = spearmanr(event_df[event_observed]['T_mono1'],
                                event_df[event_observed]['added_benefit_of_drug2'])

        r_2_delta1, _ = spearmanr(event_df[event_observed]['T_mono2'],
                                event_df[event_observed]['added_benefit_of_drug1'])

        # if the added drug is not beneficial, set correlation to nan
        if not (coxph_df_strict.at[i, 'HR1'] < 1 and coxph_df_strict.at[i, 'p1'] < 0.05):
            r_1_delta2 = np.nan
        if not (coxph_df_strict.at[i, 'HR2'] < 1 and coxph_df_strict.at[i, 'p2'] < 0.05):
            r_2_delta1 = np.nan

        r_list.append([r_bestavgres, r_t12, r_1_delta2, r_2_delta1])

    r_df = pd.DataFrame(r_list, index=coxph_df_strict.index,
                        columns=['r_bestavgres', 'r_t12', 'r_1_delta2', 'r_2_delta1'])
    merged = pd.concat([coxph_df_strict, r_df], axis=1)
    return merged


def pairplot_corr_differences(data: pd.DataFrame) -> plt.figure:
    g = sns.pairplot(data[['r_bestavgres', 'r_t12', 'r_1_delta2', 'r_2_delta1']],
                     corner=True, kind='reg', height=1.5, aspect=1, 
                     plot_kws={'scatter_kws': {'s': 3}})
    g.set(xlim=(-0.8, 0.7), ylim=(-0.8, 0.7))
    return g.fig


def compute_rmse_in_survival(true_benefit: pd.Series, inferred_benefit: pd.DataFrame) -> np.array:
    delta_t = inferred_benefit['delta_t'].sort_values(ascending=True).values
    inferred_survival = np.linspace(0, 100, delta_t.size)
    inferred = pd.DataFrame({'survival': inferred_survival, 'delta_t': delta_t})
    f_inferred = interpolate(inferred, x='delta_t', y='survival')
    
    true_delta_t = true_benefit.sort_values(ascending=True)
    true_survival = np.linspace(0, 100, true_delta_t.size)
    true = pd.DataFrame({'survival': true_survival, 'delta_t': true_delta_t})
    f_true = interpolate(true, x='delta_t', y='survival')

    t = np.linspace(0, 100, 100)
    
    return np.sqrt(np.mean(np.square(f_true(t) - f_inferred(t))))


def find_closest_benefit_to_corr(sb: SurvivalBenefit, r: float) -> SurvivalBenefit:
    sb.compute_benefit(prob_coef=50)
    highest_corr = sb.corr_rho
    if r > highest_corr:
        return sb
    sb.compute_benefit(prob_coef=-50)
    lowest_corr = sb.corr_rho
    if r < lowest_corr:
        return sb
    sb.compute_benefit_at_corr(r)
    return sb


def compute_benefits_helper(event_df: pd.DataFrame, drug1_name: str, drug2_name: str, 
                            effect_drug_idx: int, r: float) -> tuple[float, pd.DataFrame, float, pd.DataFrame]:
    km_comb = get_km_curve(event_df, 'T_comb', 'E_comb')
    if effect_drug_idx == 1:
        km_mono = get_km_curve(event_df, 'T_mono2', 'E_mono2')
        sb = SurvivalBenefit(km_mono, km_comb, 
                             drug1_name, drug1_name + '-' +  drug2_name,  
                             save_mode=False)
    elif effect_drug_idx == 2:
        km_mono = get_km_curve(event_df, 'T_mono1', 'E_mono1')
        sb = SurvivalBenefit(km_mono, km_comb, 
                             drug2_name, drug2_name + '-' +  drug1_name,  
                             save_mode=False)
    else:
        print("wrong effect_drug_idx")
        return
    
    sb.set_tmax(100)

    # correlation from bestAvgResponse
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error", UserWarning)
        try:
            sb.compute_benefit_at_corr(r)
        except UserWarning:
            sb = find_closest_benefit_to_corr(sb, r)
    
    actual_corr = sb.corr_rho
    actual_corr_benefit = sb.benefit_df.copy()
    
    # highest corr
    sb.compute_benefit(prob_coef=50)
    high_corr = sb.corr_rho
    high_corr_benefit = sb.benefit_df.copy()

    return (actual_corr, actual_corr_benefit, 
            high_corr, high_corr_benefit)


def compute_rmse_in_time(true_benefit: pd.Series, inferred_benefit: pd.DataFrame) -> np.array:
    delta_t = inferred_benefit['delta_t'].sort_values(ascending=False).values
    inferred_survival = np.linspace(0, 100, delta_t.size)
    inferred = pd.DataFrame({'Survival': inferred_survival, 'Time': delta_t})
    f = interpolate(inferred, x='Survival', y='Time')
    
    true_benefit[true_benefit < 0] = 0  # disregard negative values
    true_benefit = np.sort(true_benefit)[::-1] # descending
    surv_idx = np.linspace(0, 100 - 100 / true_benefit.size, true_benefit.size)
    
    return np.sqrt(np.mean(np.square(f(surv_idx) - true_benefit)))


def plot_actual_benefit(ori: pd.DataFrame, effect_drug_idx: str, ax=None) -> tuple[plt.figure, plt.axes]:
    # mono_idx is the effect drug monotherapy
    benefit = ori.copy()
    if effect_drug_idx == 2:
        benefit['benefit'] = benefit['T_comb'] - benefit['T_mono1']
        benefit['E_mono'] = benefit['E_mono1']
    elif effect_drug_idx == 1:
        benefit['benefit'] = benefit['T_comb'] - benefit['T_mono2']
        benefit['E_mono'] = benefit['E_mono2']

    benefit = benefit.sort_values('benefit', ascending=False)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2))
    
    n = benefit.shape[0]

    for i in range(n):
        idx = benefit.index[i]
        y_pos = 100 * i / n
        if benefit.at[idx, 'E_comb'] + benefit.at[idx, 'E_mono'] == 2:  # event observed
            ax.plot([0, benefit.at[idx, 'benefit']], [y_pos, y_pos],
                    linewidth=3, color='k', alpha=0.7, solid_capstyle='butt')

        # right censored (arrow)
        elif benefit.at[idx, 'E_comb'] == 0 and benefit.at[idx, 'E_mono'] == 1:
            ax.plot([0, benefit.at[idx, 'benefit']], [y_pos, y_pos],
                    linewidth=3, color='k', alpha=0.7, solid_capstyle='butt')
            ax.plot(benefit.at[idx, 'benefit'], y_pos, marker=9, markersize=7,
                    color='k', alpha=0.7)

        else:  # censored
            ax.plot([0, benefit.at[idx, 'benefit']], [y_pos, y_pos],
                    linewidth=3, color='gray', alpha=0.7, solid_capstyle='butt')
    ax.set_xlim(0)

    if ax is None:
        return (fig, ax)
    
    return ax


def plot_inferred_benefit(inferred_benefit: pd.DataFrame, ax=None, color='r') -> plt.axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 2))
    t = inferred_benefit['delta_t'].sort_values(ascending=True).values
    s = np.linspace(0, 100, t.size)[::-1]
    ax.plot(t, s, linewidth=3, color=color, alpha=0.9)
    return ax


def correlation_benefit_comparison(dat: pd.DataFrame) -> tuple[pd.DataFrame, plt.figure]:
    result_list = []
    coxph_df_strict = coxph_combo_vs_mono(dat, strict_censoring=True)
    event_df_dict_strict = make_event_dataframe_dict_combo(dat, coxph_df_strict, strict_censoring=True)
    info = dat[['Model', 'Tumor Type', 'BRAF_mut', 'RAS_mut']].drop_duplicates().set_index('Model')

    fig, axes = plt.subplots(5, 4, figsize=(9, 9))
    axes = axes.flatten()
    colors = sns.color_palette()
    ax_idx = 0
    
    for i in coxph_df_strict.index:
        tumor = coxph_df_strict.at[i, 'Tumor Type']
        combo = coxph_df_strict.at[i, 'Combination']
        drug1 = combo.split(' + ')[0]
        drug2 = combo.split(' + ')[1]

        event_df = event_df_dict_strict[(tumor, combo)]
        corr_dat = get_pdx_corr_data(
            dat, info, drug1, drug2, metric='BestAvgResponse')
        r, p = spearmanr(corr_dat[drug1], corr_dat[drug2])

        # Effect of drug 2
        if coxph_df_strict.at[i, 'HR1'] < 1 and coxph_df_strict.at[i, 'p1'] < 0.05:
            drug2_effect_tuple = compute_benefits_helper(
                event_df, drug1, drug2, 2, r)
            true_benefit = event_df['T_comb'] - event_df['T_mono1']
            actual_rmse = compute_rmse_in_time(true_benefit, drug2_effect_tuple[1])
            high_rmse = compute_rmse_in_time(
                true_benefit, drug2_effect_tuple[3])
            result_list.append([tumor, combo, drug2, r,
                                drug2_effect_tuple[0], drug2_effect_tuple[2],
                                actual_rmse, high_rmse])
            ax = axes[ax_idx]
            ax_idx += 1
            ax = plot_actual_benefit(event_df, 2, ax=ax)
            ax = plot_inferred_benefit(
                drug2_effect_tuple[1], ax=ax, color=colors[0])
            ax = plot_inferred_benefit(
                drug2_effect_tuple[3], ax=ax, color=colors[1])
            ax.set_title(f'Effect of {drug2}\nin {tumor} {combo}')

        # Effect of drug 1
        if coxph_df_strict.at[i, 'HR2'] < 1 and coxph_df_strict.at[i, 'p2'] < 0.05:
            drug1_effect_tuple = compute_benefits_helper(
                event_df, drug1, drug2, 1, r)
            true_benefit = event_df['T_comb'] - event_df['T_mono2']
            actual_rmse = compute_rmse_in_time(true_benefit, drug1_effect_tuple[1])
            high_rmse = compute_rmse_in_time(true_benefit, drug1_effect_tuple[3])
            result_list.append([tumor, combo, drug1, r,
                                drug1_effect_tuple[0], drug1_effect_tuple[2],
                                actual_rmse, high_rmse])
            ax = axes[ax_idx]
            ax_idx += 1
            ax = plot_actual_benefit(event_df, 1, ax=ax)
            ax = plot_inferred_benefit(
                drug1_effect_tuple[1], ax=ax, color=colors[0])
            ax = plot_inferred_benefit(
                drug1_effect_tuple[3], ax=ax, color=colors[1])
            ax.set_title(f'Effect of {drug1}\nin {tumor} {combo}')

    columns = ['Tumor Type', 'Combination', 'Target', 'r_BestAvgRes',
            'r_Actual', 'r_Highest',
            'RMSE_r_Actual', 'RMSE_r_Highest']

    result_df = pd.DataFrame(result_list, columns=columns)
    return result_df, fig


def plot_correlation_benefit_comparison_boxplot(result_df: pd.DataFrame) -> plt.figure:
    stat, p = wilcoxon(result_df['RMSE_r_Actual'], result_df['RMSE_r_Highest'])
    
    actual_df = result_df[['Tumor Type', 'Combination',
                           'Target', 'r_Actual', 'RMSE_r_Actual']]

    actual_df.loc[:, 'Correlation'] = 'Actual'
    high_df = result_df[['Tumor Type', 'Combination',
                        'Target', 'r_Highest', 'RMSE_r_Highest']]
    high_df.loc[:, 'Correlation'] = 'Highest'

    cols = ['Tumor Type', 'Combination', 'Target', 'r', 'RMSE', 'Correlation']
    actual_df.columns = cols
    high_df.columns = cols

    melted = pd.concat([actual_df, high_df], axis=0, ignore_index=False)
    fig, ax = plt.subplots(figsize=(2, 3))
    sns.boxplot(data=melted, x='Correlation', y='RMSE', ax=ax)
    ax.set_title(f'two-sided Wilcoxon test (p={p:.2f})')
    return fig


def main():
    config = load_config()['PDX']
    data_dir = config['data_dir']
    fig_dir = config['fig_dir']
    table_dir = config['table_dir']
    dat = pd.read_csv(f'{data_dir}/PDXE_drug_response.csv', 
                      header=0, index_col=None)
    dat.loc[:, 'TimeToDouble'] = dat['TimeToDouble'].round(2)

    fig1_data = prepare_dataframe_for_stripplot(dat)
    fig1 = stripplot_added_benefit(fig1_data)
    
    fig1_fname = 'PDXE_combo_added_benefit_stripplot'
    fig1_data.to_csv(f'{table_dir}/{fig1_fname}.source_data.csv', index=False)
    fig1.savefig(f'{fig_dir}/{fig1_fname}.pdf', bbox_inches='tight')

    added_benefit_data_tuple = prepare_dataframes_for_distplot(dat)
    fig2 = distplot_monotherapy_and_combo_added_benefit(
        *added_benefit_data_tuple)
    fig3 = cumulative_distplot_monotherapy_and_combo_added_benefit(
        *added_benefit_data_tuple)

    fig2.savefig(f'{fig_dir}/PDXE_added_benefit_distplot.pdf', bbox_inches='tight')
    fig3.savefig(f'{fig_dir}/PDXE_added_benefit_cumulative_distplot.pdf', bbox_inches='tight')

    merged = merge_dataframes(*added_benefit_data_tuple)
    merged.to_csv(f'{table_dir}/PDXE_added_benefit_distplot.source_data.csv', index=False)

    fig4_data = corr_AB_vs_corr_A_deltaB(dat)
    fig4 = pairplot_corr_differences(fig4_data)
    fig4.savefig(f'{fig_dir}/PDXE_corr_differences_pairplot.pdf', bbox_inches='tight')
    fig4_data.to_csv(f'{table_dir}/PDXE_corr_differences_pairplot.source_data.csv', index=False)

    fig56_data, fig5 = correlation_benefit_comparison(dat)
    fig5.savefig(f'{fig_dir}/PDXE_actual_vs_high_corr_benefit_profiles.pdf', bbox_inches='tight')
    
    fig6 = plot_correlation_benefit_comparison_boxplot(fig56_data)
    fig6.savefig(f'{fig_dir}/PDXE_actual_vs_high_corr_benefit_boxplot.pdf', bbox_inches='tight')
    fig56_data.to_csv(f'{table_dir}/PDXE_actual_vs_high_corr_benefit_boxplot.source_data.csv', index=False)


if __name__ == '__main__':
    main()