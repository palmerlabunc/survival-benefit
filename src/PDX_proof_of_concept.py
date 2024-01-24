import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon
from lifelines import KaplanMeierFitter
from typing import Tuple
from PDXE_correlation import get_pdx_corr_data
from utils import load_config, set_figure_size_dim, get_xticks
from PDX_proof_of_concept_helper import *
import warnings

warnings.filterwarnings("ignore")

# Three main analyses:
# 1. Is there anatognism in combination threapies?
# 2. How much difference is there betweeen corr(A, deltaB) and corr(A, B)?
# 3. Does using corr(A, B) better infer the benefit profile than using high correlation?

config = load_config()
DELTA_T = config['PDX']['delta_t']
COLOR_DICT = config['colors']

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


def stripplot_added_benefit(data: pd.DataFrame) -> plt.Figure:
    data.loc[:, 'label'] = 'Effect of ' + \
        data['target'] + ' in ' + data['Combination']
    g = sns.catplot(y="label", x="added_benefit", hue="event_observed",
                    data=data, kind="strip", height=4, aspect=1.5)
    g.fig.axes[0].axvline(0, linestyle='--', color='k')
    plt.xlabel('Added benefit (days)')
    plt.ylabel('')
    return g.fig


def prepare_dataframes_for_distplot(dat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comb_active = prepare_dataframe_for_stripplot(dat)

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

    return mono_merged, active, inactive, comb_active


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
                                                            mono_inactive: pd.DataFrame, combo_active: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(2, 2))
    sns.despine()

    sns.ecdfplot(x='added_benefit', data=mono_merged,
                color=sns.color_palette()[0], ax=ax)
    sns.ecdfplot(x='added_benefit', data=mono_active,
                color=sns.color_palette()[1], ax=ax)
    sns.ecdfplot(x='added_benefit', data=mono_inactive,
                color=sns.color_palette()[2], ax=ax)
    sns.ecdfplot(x='added_benefit', data=combo_active,
                color=sns.color_palette()[3], ax=ax)
    ax.set_xlabel('added benefit (day)')
    ax.set_ylabel('Cumulative density')
    ax.set_xlim(-100, 0)
    ax.axvline(0, linestyle='--', color='k', alpha=0.5)
    ax.legend(labels=['All monotherapy', 'Active monotherapy',
             'Inactive monotherapy', 'Successful combination'],
              bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig


def merge_dataframes(mono_merged: pd.DataFrame, mono_active: pd.DataFrame,
                     mono_inactive: pd.DataFrame, combo_active: pd.DataFrame) -> pd.DataFrame:
    combo_active.loc[:, 'Treatment'] = combo_active['Combination']
    mono_active.loc[:, 'Activity'] = 'active'
    mono_inactive.loc[:, 'Activity'] = 'inactive'
    merged = pd.concat([mono_active, mono_inactive, combo_active], axis=0,
                       ignore_index=True, join='outer')
    return merged


def compute_r_A_deltaB(event_df: pd.DataFrame, control_drug_idx: int, tmax=100) -> float:
    event_df_tmp = event_df.copy()
    event_df_tmp.loc[event_df_tmp['E_mono1'] == 0, 'T_mono1'] = tmax
    event_df_tmp.loc[event_df_tmp['E_mono2'] == 0, 'T_mono2'] = tmax

    event_observed = event_df_tmp['E_comb'] + event_df_tmp[f'E_mono{control_drug_idx}'] == 2
    deltaB =  (event_df_tmp['T_comb'] - event_df_tmp[f'T_mono{control_drug_idx}'])
    r, p = spearmanr(event_df_tmp[event_observed][f'T_mono{control_drug_idx}'],
                     deltaB[event_observed])
    return r


def corr_AB_vs_corr_A_deltaB(dat: pd.DataFrame) -> pd.DataFrame:
    tmax = 100
    coxph_df = coxph_combo_vs_mono(dat, strict_censoring=False)
    event_df_dict = make_event_dataframe_dict_combo(dat, coxph_df, 
                                                    strict_censoring=False)
    info = dat[['Model', 'Tumor Type', 'BRAF_mut', 'RAS_mut']].drop_duplicates().set_index('Model')
    
    r_list = []
    
    for i in coxph_df.index:
        tumor = coxph_df.at[i, 'Tumor Type']
        combo = coxph_df.at[i, 'Combination']
        drug1 = combo.split(' + ')[0]
        drug2 = combo.split(' + ')[1]

        event_df = event_df_dict[(tumor, combo)].copy()
        corr_dat = get_pdx_corr_data(
            dat, info, drug1, drug2, metric='BestAvgResponse')
        r_bestavgres, _ = spearmanr(corr_dat[drug1], corr_dat[drug2])

        # Effect of drug 2
        if coxph_df.at[i, 'HR1'] < 1 and coxph_df.at[i, 'p1'] < 0.05:
            r = compute_r_A_deltaB(event_df, 2, tmax=tmax)
            r_list.append([tumor, combo, drug2, r_bestavgres, r])

        # Effect of drug 1
        if coxph_df.at[i, 'HR2'] < 1 and coxph_df.at[i, 'p2'] < 0.05:
            r = compute_r_A_deltaB(event_df, 1, tmax=tmax)
            r_list.append([tumor, combo, drug2, r_bestavgres, r])

    columns = ['Tumor Type', 'Combination',
            'Effect Drug', 'r_BestAvgRes', 'r_A_vs_deltaB']

    r_df = pd.DataFrame(r_list,
                        columns=columns)
    
    return r_df


def plot_corr_AB_vs_corr_A_deltaB(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(x='r_BestAvgRes', y='r_A_vs_deltaB', data=data, ax=ax)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.plot([-1, 1], [-1, 1], color='k', linestyle='--')
    return fig


def plot_benefits(true_benefit_event_df: pd.DataFrame,
                  ax: plt.Axes,
                  color_dict: dict,
                  inferred_deltat: np.array = None,
                  visual_deltat: np.array = None,
                  tmax: float = None) -> plt.Axes:
    
    true_benefit_km = KaplanMeierFitter()
    true_benefit_km.fit(true_benefit_event_df['T_benefit'], 
                        true_benefit_event_df['E_benefit'])
    
    inferred_benefit_km = KaplanMeierFitter()
    inferred_benefit_km.fit(inferred_deltat)

    if tmax is None:
        tmax = true_benefit_event_df['T_benefit'].max()
    ts = np.linspace(0, tmax, 500)

    ax.plot(ts, 100 * true_benefit_km.survival_function_at_times(ts),
            color='k', linewidth=1.5, label='True benefit of indvidual PDXs')
    
    ax.plot(ts, 100 * inferred_benefit_km.survival_function_at_times(ts),
            color=color_dict['inference'], linewidth=1.2, 
            label='Inferred benefit')
    
    if visual_deltat is not None:
        visual_benefit_km = KaplanMeierFitter()
        visual_benefit_km.fit(visual_deltat)
        ax.plot(ts, 100 * visual_benefit_km.survival_function_at_times(ts),
                color=color_dict['visual_appearance'], linewidth=1.2, 
                label='Apparent benefit')
    
    ax.set_xlim(0, tmax)
    ax.set_xticks(get_xticks(tmax, metric='days'))
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel('Added benefit (days)')
    ax.set_ylabel('PDX models (%)')

    return ax


def correlation_benefit_comparison(dat: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    result_list = []
    coxph_df = coxph_combo_vs_mono(dat, strict_censoring=False)
    event_df_dict = make_event_dataframe_dict_combo(dat, coxph_df, strict_censoring=False)
    info = dat[['Model', 'Tumor Type', 'BRAF_mut', 'RAS_mut']].drop_duplicates().set_index('Model')

    n_combos = coxph_df.shape[0]
    fig, axes = set_figure_size_dim(n_combos, 
                                    ax_width=1.5, ax_height=1.5, max_cols=4)
    ax_idx = 0

    for i in coxph_df.index:
        tumor = coxph_df.at[i, 'Tumor Type']
        combo = coxph_df.at[i, 'Combination']
        drug1 = combo.split(' + ')[0]
        drug2 = combo.split(' + ')[1]

        event_df = event_df_dict[(tumor, combo)]

        for control_drug_idx in [1, 2]:
            if coxph_df.at[i, f'HR{control_drug_idx}'] >=1 or coxph_df.at[i, f'p{control_drug_idx}'] >= 0.05:
                continue
            
            # compute true benefit
            event_df = event_df_dict[(tumor, combo)]
            benefit_event_df = added_benefit_event_table(event_df, control_drug_idx)
            n = benefit_event_df.shape[0]

            # use only models that have observed events in the monotherapy arm
            # to calculate correlation
            corr_dat = get_pdx_corr_data(dat, info, drug1, drug2, 
                                         metric='BestAvgResponse')
            corr_dat = corr_dat.reindex(benefit_event_df.index)
            r_bestavgres, p = spearmanr(corr_dat[drug1], corr_dat[drug2])

            event_df = event_df.reindex(benefit_event_df.index)
            r_A_deltaB = compute_r_A_deltaB(event_df, control_drug_idx)
            
            # infer benefit using BestAvgResponse correlation
            actual_r_bestavgres, r_bestavgres_benefit = compute_benefits_helper(
                event_df, drug1, drug2, control_drug_idx, r_bestavgres)
            # infer benefit using A_deltaB correlation
            actual_r_A_deltaB, r_A_deltaB_benefit = compute_benefits_helper(
                event_df, drug1, drug2, control_drug_idx, r_A_deltaB)
            # infer benefit using high correlation (visual appearance)
            actual_r_high, r_high_benefit = compute_benefits_helper(
                event_df, drug1, drug2, control_drug_idx, 1)
            
            # compute RMSE
            A_deltaB_rmse = compute_rmse_in_survival_km(benefit_event_df, 
                                                        r_A_deltaB_benefit[r_A_deltaB_benefit['valid']][DELTA_T])
            
            bestavgres_rmse = compute_rmse_in_survival_km(benefit_event_df,
                                                          r_bestavgres_benefit[r_bestavgres_benefit['valid']][DELTA_T])
            high_rmse = compute_rmse_in_survival_km(benefit_event_df,
                                                    r_high_benefit[r_high_benefit['valid']][DELTA_T])

            # plot
            ax = axes[ax_idx]
            
            if control_drug_idx == 1:
                ax.set_title(f'{tumor} {drug2} + {drug1}')
                result_list.append([tumor, combo, drug2, n, r_bestavgres, r_A_deltaB,
                                    actual_r_bestavgres, actual_r_A_deltaB, actual_r_high,
                                    bestavgres_rmse, A_deltaB_rmse, high_rmse])
            else:
                ax.set_title(f'{tumor} {drug1} + {drug2}')
                result_list.append([tumor, combo, drug1, n, r_bestavgres, r_A_deltaB,
                                    actual_r_bestavgres, actual_r_A_deltaB, actual_r_high,
                                    bestavgres_rmse, A_deltaB_rmse, high_rmse])
            
            ax = plot_benefits(benefit_event_df, ax, COLOR_DICT,
                               inferred_deltat=r_bestavgres_benefit[DELTA_T],
                               visual_deltat=r_high_benefit[DELTA_T])
            
            ax_idx += 1

    columns = ['Tumor Type', 'Combination', 'Effect Drug', 'N', 'r_BestAvgRes', 'r_A_deltaB',
               'r_BestAvgRes_Actual', 'r_A_deltaB_Actual', 'r_Highest_Actual',
               'RMSE_r_BestAvgRes', 'RMSE_r_A_deltaB', 'RMSE_r_Highest']

    result_df = pd.DataFrame(result_list, columns=columns)
    return result_df, fig


def plot_correlation_benefit_comparison_2lineplot(result_df: pd.DataFrame) -> plt.Figure:
    stat, p = wilcoxon(result_df['RMSE_r_Highest'], 
                       result_df['RMSE_r_BestAvgRes'])
    n_combo = result_df.shape[0]
    n_tumors = result_df['N'].sum()
    fig, ax = plt.subplots(figsize=(1.2, 1.5))
    for i in range(result_df.shape[0]):
        ax.plot([0, 1], 
                [result_df.at[i, 'RMSE_r_Highest'], result_df.at[i, 'RMSE_r_BestAvgRes']], 
                marker='o', color='k', 
                markersize=3, linewidth=0.75)
    ax.set_title(f'Wilcoxon p={p:.1e}\n{n_combo} combinations, {n_tumors} tumors')
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Apparent', 'Inferred'])
    ax.set_xlabel('Benefit')
    ax.set_ylabel('Error (%)')

    return fig


def bootstrapping_test_for_antagonism(combo_added_benefit: pd.DataFrame, 
                                      mono_added_benefit: pd.DataFrame, 
                                      n_rep=10000) -> Tuple[np.array, float, float]:
    rng = np.random.default_rng(0)
    antag_mean_arr = np.zeros(n_rep)
    for run in range(n_rep):
        sampled_idx = rng.integers(0, len(mono_added_benefit), size=len(combo_added_benefit))
        mono_sampled = mono_added_benefit.loc[sampled_idx, 'added_benefit']
        neg_samples = mono_sampled[mono_sampled < 0]
        antag_mean = neg_samples.sum() / neg_samples.shape[0]
        antag_mean_arr[run] = antag_mean

    neg_benefit = combo_added_benefit[combo_added_benefit['added_benefit'] < 0]
    comb_antag_mean = neg_benefit['added_benefit'].sum() / neg_benefit.shape[0]
    null_mean = antag_mean_arr.mean()
    diff = abs(null_mean - comb_antag_mean)
    p = ((antag_mean_arr < null_mean - diff).sum() + (antag_mean_arr > null_mean + diff).sum()) / n_rep
    return (antag_mean_arr, comb_antag_mean, p)


def plot_boostrapping_distribution(antag_mean_arr: np.array, 
                                   comb_antag_mean: float, 
                                   p: float, 
                                   label: str = None,
                                   ax: plt.Axes = None) -> plt.Figure | plt.Axes:
    if label == 'Active monotherapy':
        color = sns.color_palette('deep')[1]
    if label == 'Inactive monotherapy':
        color = sns.color_palette('deep')[2]
    else:
        color = sns.color_palette('deep')[0]
    
    ax_given = True
    if ax is None:
        ax_given = False
        fig, ax = plt.subplots(figsize=(3, 2))
    
    sns.histplot(antag_mean_arr, ax=ax, stat='density', 
                 label=f'Null from {label} p={p:.1e}',
                 color=color)
    ax.axvline(comb_antag_mean, 
               color='k', linestyle='--',
               label='Successful combination')
    ax.set_xlabel('Mean negative benefit (days)')
    if ax_given:
        return ax
    return fig


def mean_times_prob_of_negative_benefit(df: pd.DataFrame) -> float:
    prob = (df['added_benefit'] < 0).sum() / df.shape[0]
    avg = df[df['added_benefit'] < 0]['added_benefit'].mean()
    return round(avg * prob, 2)


def paired_test_for_antagonism(combo_added_benefit: pd.DataFrame, 
                               mono_added_benefit: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    mono_grouped = mono_added_benefit.groupby(['Tumor Type', 'Treatment'])
    comb_grouped = combo_added_benefit.groupby(['Tumor Type', 'Combination', 'target'])
    vals = []
    for idx, comb in comb_grouped:
        cancer, combo, target = idx
        mono = mono_grouped.get_group((cancer, target))
        comb_val = mean_times_prob_of_negative_benefit(comb)
        mono_val = mean_times_prob_of_negative_benefit(mono)
        vals.append([cancer, combo, target, mono_val, comb_val])

    df = pd.DataFrame(vals, columns=['Tumor Type', 'Combination',
                      'target', 'mono_added_benefit', 'combo_added_benefit'])
    stat, p = wilcoxon(df['mono_added_benefit'], df['combo_added_benefit'])
    return (df, p)


def plot_paired_test_for_antagonism_lineplot(df: pd.DataFrame, p: float) -> plt.figure:
    fig, ax = plt.subplots(figsize=(2, 2))
    for i in df.index:
        ax.plot([0, 1], [df.at[i, 'mono_added_benefit'], df.at[i, 'combo_added_benefit']],
                 color='k', alpha=0.8, marker='o')
    ax.set_xticks([0, 1])
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(f"Wilcoxon test p={p:.2f}")
    ax.set_xticklabels(['Monotherapy', 'Combination'])
    ax.set_ylabel('Mean * proportion of\nnegative added benefit (days)')
    return fig


def plot_one_combo_example_barplot(dat: pd.DataFrame, 
                                   tumor: str, control_drug: str, added_drug: str,
                                   version: int = 1) -> plt.Figure:
    combo_drug = control_drug + ' + ' + added_drug
    models = get_models(dat, tumor, combo_drug)

    event_mono = get_event_table(dat, models, control_drug, 
                                 strict_censoring=False)
    event_combo = get_event_table(dat, models, combo_drug, 
                                  strict_censoring=False)
    
    event_models = np.intersect1d(event_mono[event_mono['Event'] == 1].index,
                                  event_combo[event_combo['Event'] == 1].index)
    
    merged = pd.merge(event_mono['Time'].reindex(event_models),
                      event_combo['Time'].reindex(event_models),
                      left_index=True, right_index=True,
                      suffixes=[f'_control', '_combo'])
    
    merged = merged.sort_values('Time_combo', ascending=False)

    if tumor == 'PDAC' and added_drug == 'binimetinib' and control_drug == 'BKM120':
        example_df = merged.reindex(['X-2026', 'X-2997', 'X-2043'])
    else:
        example_df = merged.iloc[:3, :]

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    
    # Version 1: have some space inbetween for ...
    if version == 1:
        for i in range(example_df.shape[0]):
            t_control = example_df.iat[i, 0] # control
            t_combo = example_df.iat[i, 1] # combo
            
            if i > 0:
                i += 2
            ax.plot([0, t_control], [i+0.2, i+0.2], 
                    linewidth=5,
                    alpha=0.8,
                    color=COLOR_DICT['control_arm'],
                    solid_capstyle='butt')
            ax.plot([t_control, t_combo], [i+0.2, i+0.2], 
                    linewidth=5,
                    alpha=0.5,
                    color='k',
                    solid_capstyle='butt')
            ax.plot([0, t_combo], [i-0.2, i-0.2], 
                    linewidth=5,
                    color=COLOR_DICT['combination_arm'],
                    alpha=0.8,
                    solid_capstyle='butt')

        ax.set_ylim(-0.5, 4.5)
        ax.set_yticks(range(5))
        ax.set_yticklabels([example_df.index[0], ':', ':', 
                            example_df.index[1], example_df.index[2]])
    
    # Version 2: no space inbetween
    elif version == 2:
        for i in range(example_df.shape[0]):
            t_control = example_df.iat[i, 0] # control
            t_combo = example_df.iat[i, 1] # combo
            
            ax.plot([0, t_control], [i+0.15, i+0.15], 
                    linewidth=7,
                    color=COLOR_DICT['control_arm'],
                    alpha=0.8,
                    label=control_drug,
                    solid_capstyle='butt')
            ax.plot([t_control, t_combo], [i+0.15, i+0.15], 
                    linewidth=7,
                    color='k',
                    alpha=0.5,
                    label=f'benefit of {added_drug}',
                    solid_capstyle='butt')
            ax.plot([0, t_combo], [i-0.15, i-0.15], 
                    linewidth=7,
                    color=COLOR_DICT['combination_arm'],
                    alpha=0.8,
                    label=f'{control_drug} + {added_drug}',
                    solid_capstyle='butt')
            ax.set_ylim(-0.5, 2.5)
        ax.set_yticks(range(3))
        ax.set_yticklabels(example_df.index)

    tmax = example_df['Time_combo'].max()
    ax.set_xlim(0, tmax)
    ax.set_xticks(get_xticks(tmax, metric='days'))
    ax.set_xlabel('Time to double (days)')
    ax.set_title(f'{tumor} {added_drug} + {control_drug}')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig


def test_antagonism_and_correlation(dat: pd.DataFrame, pdx_config: Mapping):
    data_dir = pdx_config['data_dir']
    fig_dir = pdx_config['fig_dir']
    table_dir = pdx_config['table_dir']

    # Figure 2 - distplot of added benefit
    mono_merged, mono_active, mono_inactive, comb_active = prepare_dataframes_for_distplot(dat)
    fig2 = distplot_monotherapy_and_combo_added_benefit(
        mono_merged, mono_active, mono_inactive, comb_active)
    
    fig2.savefig(f'{fig_dir}/PDXE_added_benefit_distplot.pdf', 
                 bbox_inches='tight')
    
    # Figure 3 - cumulative distplot of added benefit
    fig3 = cumulative_distplot_monotherapy_and_combo_added_benefit(
        mono_merged, mono_active, mono_inactive, comb_active)

    fig3.savefig(f'{fig_dir}/PDXE_added_benefit_cumulative_distplot.pdf', 
                 bbox_inches='tight')
    
    # Testing for antagonism - paired test
    fig7_data, p = paired_test_for_antagonism(comb_active, mono_active)
    fig7 = plot_paired_test_for_antagonism_lineplot(fig7_data, p)
    fig7.savefig(f'{fig_dir}/PDXE_paired_test_for_antagonism_lineplot.pdf', 
                 bbox_inches='tight')
    fig7_data.to_csv(f'{table_dir}/PDXE_paired_test_for_antagonism_lineplot.source_data.csv', 
                     index=False)

    # Testing for antagonism - bootstrapping test
    active_antag_mean_arr, comb_antag_mean, p_active= bootstrapping_test_for_antagonism(comb_active, 
                                                                               mono_active)
    inactive_antag_mean_arr, comb_antag_mean, p_inactive = bootstrapping_test_for_antagonism(comb_active, 
                                                                                  mono_inactive)
    fig8, ax = plt.subplots(figsize=(2, 2))

    ax = plot_boostrapping_distribution(active_antag_mean_arr, comb_antag_mean, p_active,
                                        label='Active monotherapy',
                                        ax=ax)
    ax = plot_boostrapping_distribution(inactive_antag_mean_arr, comb_antag_mean, p_inactive,
                                        label='Inactive monotherapy',
                                        ax=ax)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig8.savefig(f'{fig_dir}/PDXE_bootstrapping_test_for_antagonism.pdf', 
                 bbox_inches='tight')

    # Figure 1 - stripplot of added benefit
    fig1_data = prepare_dataframe_for_stripplot(dat)
    fig1 = stripplot_added_benefit(fig1_data)
    
    fig1_fname = 'PDXE_combo_added_benefit_stripplot'
    fig1_data.to_csv(f'{table_dir}/{fig1_fname}.source_data.csv', index=False)
    fig1.savefig(f'{fig_dir}/{fig1_fname}.pdf', bbox_inches='tight')


    # Figure 4 - scatterplot of correlation differences
    merged = merge_dataframes(mono_merged, mono_active, mono_inactive, comb_active)
    merged.to_csv(f'{table_dir}/PDXE_added_benefit_distplot.source_data.csv', 
                  index=False)

    fig4_data = corr_AB_vs_corr_A_deltaB(dat)
    fig4 = plot_corr_AB_vs_corr_A_deltaB(fig4_data)
    fig4.savefig(f'{fig_dir}/PDXE_corr_differences_scatterplot.pdf', 
                 bbox_inches='tight')
    fig4_data.to_csv(f'{table_dir}/PDXE_corr_differences_scatterplot.source_data.csv', 
                     index=False)
    

def one_combo_example(dat: pd.DataFrame, pdx_config: Mapping):
    """Plot and save figures for one combination example

    Args:
        dat (pd.DataFrame): _description_
        pdx_config (Mapping): _description_
    """
    fig_dir = pdx_config['fig_dir']
    # one combination example: PDAC binimetinib + BKM120
    cond1 = dat['Treatment'].isin([pdx_config['example_experimental'], 
                                   pdx_config['example_control'], 
                                   pdx_config['example_combo']])
    cond2 = dat['Tumor Type'] == pdx_config['example_tumor']
    one_combo_data = dat[cond1 & cond2]
    _, one_combo_benefit_fig = correlation_benefit_comparison(one_combo_data)
    one_combo_benefit_fig.savefig(f'{fig_dir}/PDXE_actual_vs_high_corr_benefit_profiles_one_combo.pdf',
                                  bbox_inches='tight')

    fig_barplot1 = plot_one_combo_example_barplot(dat, pdx_config['example_tumor'], 
                                                  pdx_config['example_control'], 
                                                  pdx_config['example_experimental'], 
                                                  version=1)
    fig_barplot1.savefig(f'{fig_dir}/PDXE_actual_vs_high_corr_benefit_profiles_one_combo_barplot1.pdf',
                         bbox_inches='tight')
    
    fig_barplot2 = plot_one_combo_example_barplot(dat, pdx_config['example_tumor'],
                                                  pdx_config['example_control'], 
                                                  pdx_config['example_experimental'], 
                                                  version=2)
    fig_barplot2.savefig(f'{fig_dir}/PDXE_actual_vs_high_corr_benefit_profiles_one_combo_barplot2.pdf',
                         bbox_inches='tight')


def compare_inferred_apparent_to_actual_benefit(dat: pd.DataFrame, pdx_config: Mapping):
    fig_dir = pdx_config['fig_dir']
    table_dir = pdx_config['table_dir']

    # Figure 5 - actual benefit vs. inferred benefit
    fig56_data, fig5 = correlation_benefit_comparison(dat)
    fig5.savefig(f'{fig_dir}/PDXE_actual_vs_high_corr_benefit_profiles.pdf', 
                 bbox_inches='tight')
    
    fig6_2lineplot = plot_correlation_benefit_comparison_2lineplot(fig56_data)
    fig6_2lineplot.savefig(f'{fig_dir}/PDXE_actual_vs_high_corr_{DELTA_T}_benefit_2lineplot.pdf', 
                           bbox_inches='tight')
    
    fig56_data.to_csv(f'{table_dir}/PDXE_actual_vs_high_corr_{DELTA_T}_benefit_3lineplot.source_data.csv',
                    index=False)

    # Figure 6 - boxplot/lineplot actual benefit vs. inferred benefit RMSE by correlation
    #fig6_3lineplot = plot_correlation_benefit_comparison_3lineplot(fig56_data)
    #fig6_3lineplot.savefig(f'{fig_dir}/PDXE_actual_vs_high_corr_{DELTA_T}_benefit_3lineplot.pdf', 
    #                       bbox_inches='tight')

    #fig6_3boxplot = plot_correlation_benefit_comparison_3boxplot(fig56_data)
    #fig6_3boxplot.savefig(f'{fig_dir}/PDXE_actual_vs_high_corr_{DELTA_T}_benefit_3boxplot.pdf',
    #                      bbox_inches='tight')

    #fig6_2boxplot = plot_correlation_benefit_comparison_2boxplot(fig56_data)
    #fig6_2boxplot.savefig(f'{fig_dir}/PDXE_actual_vs_high_corr_{DELTA_T}_benefit_2boxplot.pdf',
    #                      bbox_inches='tight')



def main():
    plt.style.use('env/publication.mplstyle')

    pdx_config = load_config()['PDX']
    data_dir = pdx_config['data_dir']
    
    dat = pd.read_csv(f'{data_dir}/PDXE_drug_response.csv', 
                      header=0, index_col=None)
    dat.loc[:, 'TimeToDouble'] = dat['TimeToDouble'].round(2)

    test_antagonism_and_correlation(dat, pdx_config)
    one_combo_example(dat, pdx_config)
    compare_inferred_apparent_to_actual_benefit(dat, pdx_config)


if __name__ == '__main__':
    main()