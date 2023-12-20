from pathlib import Path
from multiprocessing import Pool
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import glob
from survival_benefit.survival_benefit_class import SurvivalBenefit
from utils import load_config


def endpoint_simple(endpoint: str) -> str:
    if endpoint == 'OS':
        return 'OS'
    return 'Surrogate'


def compile_stats(table_dir: str, corr_type: str) -> pd.DataFrame:
    stat_df = pd.DataFrame()
    
    for combo in os.listdir(table_dir):
        dirname = os.path.join(table_dir, combo)
        if not os.path.isdir(dirname):
            continue
        if len(os.listdir(dirname)) == 0:
            continue
        #FIXME this is a work-around. Make sure we used the specified target corr
        print(dirname)

        if corr_type == "high":
            benefit_filename = glob.glob(
                f'{dirname}/N_500.target_corr_1.00.*.table.csv')[0]
            stats_filename = glob.glob(
                f'{dirname}/N_500.target_corr_1.00.*.summary_stats.csv')[0]
        else:
            try:
                benefit_filename = glob.glob(
                    f'{dirname}/N_500.target_corr_0.*.table.csv')[0]
            except IndexError:
                benefit_filename = glob.glob(
                    f'{dirname}/N_500.target_corr_-0.*.table.csv')[0]
            try:
                stats_filename = glob.glob(
                    f'{dirname}/N_500.target_corr_0.*.summary_stats.csv')[0]
            except IndexError:
                stats_filename = glob.glob(
                    f'{dirname}/N_500.target_corr_-0.*.summary_stats.csv')[0]
        
        stats = pd.read_csv(stats_filename,
                            index_col=0, header=None).squeeze(axis=1)
        benefit_df = pd.read_csv(
            benefit_filename, index_col=0, header=0).squeeze(axis=1)
        
        combo = stats['Combination']
        if 'VanCutsem2015' in combo:
            tokens = combo.rsplit('_', 2)
            stats.loc['combo_name'] = tokens[0]
            stats.loc['endpoint'] = endpoint_simple(tokens[1])

        else:
            tokens = combo.rsplit('_', 1)
            stats.loc['combo_name'] = tokens[0]
            stats.loc['endpoint'] = endpoint_simple(tokens[1])
        
        stat_df = pd.concat([stat_df, stats], axis=1, join='outer')
    
    stat_df = stat_df.T
    #FIXME I don't know why this line isn't working
    stat_df = stat_df.apply(pd.to_numeric, errors='ignore', axis=1)
    
    stat_df.loc[:, 'Gini_coefficient'] = stat_df['Gini_coefficient'].astype(float)
    stat_df.loc[:, 'Percent_patients_benefitting_1mo_from_valid_highbound'] = stat_df['Percent_patients_benefitting_1mo_from_valid_highbound'].astype(float)
    stat_df.loc[:, 'Median_benefit_highbound'] = stat_df['Median_benefit_highbound'].astype(float)
    return stat_df


def merge_with_metadata(metadata: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.dropna(subset=['Key', 'Indication'])
    metadata.drop_duplicates(
        subset=['Cancer Type', 'Experimental', 'Control', 'Trial ID', 'Trial Name'],
        inplace=True)

    merged = pd.merge(metadata, stats, left_on='Key', right_on='combo_name', how='right')
    return merged


def plot_1mo_responder_percentage(stats: pd.DataFrame, highlight=False) -> plt.figure:
    col = 'Percent_patients_benefitting_1mo_from_valid_highbound'
    stat_df = stats.sort_values(col).reset_index(drop=True)
    stat_df.loc[:, 'above_50'] = stat_df[col] > 50
    color_dict = stat_df['above_50'].map(
        {True: sns.color_palette('deep')[0],
        False: sns.color_palette('deep')[1]})

    fig, ax = plt.subplots(figsize=(8, 3))
    if highlight:
        stat_df.plot.bar(y=col, color=color_dict, ax=ax)
        ax.axhline(50, color='black', linestyle='--')
    else:
        stat_df.plot.bar(y=col, ax=ax)

    ax.set_xticklabels('')
    ax.set_ylabel('Patients benefitting\nat least 1 month (%)')
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel('Combination therapies')
    ax.set_ylim(0, 100)
    ax.get_legend().remove()
    return fig


def plot_median_benefit_simulated_vs_actual(metadata: pd.DataFrame, stats: pd.DataFrame, 
                                            highlight=False, endpoint: str = None) -> plt.figure:
    data_master_df = metadata.dropna(subset=['Key', 'Indication'])
    data_master_df.drop_duplicates(
        subset=['Cancer Type', 'Experimental', 'Control', 'Trial ID', 'Trial Name'],
        inplace=True)
    actual_median_benefit_df = pd.concat([pd.Series(index=data_master_df['Key'] + '_OS',
                                                data=data_master_df['OS median benefit'].values),
                                      pd.Series(index=(data_master_df['Key'] + '_' + data_master_df['Surrogate Metric']).values,
                                                data=data_master_df['Surrogate median benefit'].values).dropna()],
                                     axis=0)
    actual_median_benefit_df = actual_median_benefit_df[~actual_median_benefit_df.isin(['Not reached for comb', 'Not reached'])].astype(float)
    actual_median_benefit_df.dropna(inplace=True)
    actual_median_benefit_df = actual_median_benefit_df.to_frame(name='actual')

    merged = pd.merge(actual_median_benefit_df, stats, 
                      left_index=True, right_on='Combination', 
                      how='right')
    
    merged.loc[:, '3mo_more_benefit'] = merged['Median_benefit_highbound'] - merged['actual'] >= 3
    if endpoint is not None:
        merged = merged[merged['endpoint'] == endpoint]

    fig, ax = plt.subplots(figsize=(3,3))
    if not highlight:
        sns.scatterplot(y='Median_benefit_highbound', 
                        x='actual', 
                        style='endpoint', 
                        data=merged, 
                        ax=ax, color=sns.color_palette('deep')[3])
    else:
        sns.scatterplot(y='Median_benefit_highbound', x='actual', style='endpoint',
                        hue='3mo_more_benefit',
                        hue_order=[False, True],
                        data=merged, 
                        ax=ax, palette=sns.color_palette('deep'))
    ax.plot([0, 27], [0, 27], color='black', linestyle='--')
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 27)
    ax.set_xticks([0, 10, 20])
    ax.set_yticks([0, 10, 20])
    ax.set_ylabel('Simulated median benefit\n(months)')
    ax.set_xlabel('median benefit reported in trial\n(months)')
    return fig


def plot_gini_histplot(compiled_stats: pd.DataFrame) -> plt.figure:
    sns.set_palette('deep')
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.histplot(x=compiled_stats['Gini_coefficient'], binwidth=0.1,
                 binrange=(0, 1),
                 color=sns.color_palette('deep')[4],
                 ax=ax)
    ax.set_xlabel('Gini coefficient')
    ax.set_xlim(0, 1)

    return fig


def plot_gini_compare_experimental_and_high(exp_corr_stats: pd.DataFrame,
                                            high_orr_stats: pd.DataFrame) -> plt.figure:
    fig, ax = plt.subplots(figsize=(3,2))
    merged = pd.merge(exp_corr_stats, high_orr_stats, on=['Combination', 'Monotherapy'],
                      suffixes=['_exp', '_high'])
    sns.set_palette('deep')
    sns.kdeplot(x='Gini_coefficient_high', data=merged,
                label='Highest correlation',
                fill=True, alpha=0.5, linewidth=0,
                ax=ax)
    sns.kdeplot(x='Gini_coefficient_exp', data=merged, 
                label='Experimental correlation',
                fill=True, alpha=0.5, linewidth=0,
                ax=ax)
    ax.set_xlabel('Gini coefficient')
    ax.legend()
    ax.set_xlim(0, 1)
    return fig


def plot_gini_by_something_boxplot(compiled_stats: pd.DataFrame, by: str, n_min=5) -> plt.figure:
    """_summary_

    Args:
        compiled_stats (pd.DataFrame): _description_
        by (str): categorize by this column
        n_min (int, optional): At least n_min number of points to show up on boxplot. Defaults to 5.

    Returns:
        plt.figure:
    """
    sns.set_palette('deep')
    fig, ax = plt.subplots(figsize=(3, 2))
    dat = compiled_stats[compiled_stats.groupby(by)[by].transform('size') > n_min]
    sns.boxplot(y=by, x='Gini_coefficient', data=dat, ax=ax)
    ax.set_ylabel(by)
    ax.set_xlabel('Gini coefficient')
    ax.set_xlim(0, 1)
    return fig


def main():
    plt.style.use('env/publication.mplstyle')
    config = load_config()
    high_corr_stats = compile_stats(config['main_combo']['table_dir'] + '/predictions', 'high')
    exp_corr_stats = compile_stats(config['main_combo']['table_dir'] + '/predictions', 'exp')
    metadata = pd.read_excel(config['data_master_sheet'], 
                             engine='openpyxl')
    extended_exp_corr_stats = merge_with_metadata(metadata, exp_corr_stats)

    high_corr_stats.to_csv(f"{config['main_combo']['table_dir']}/high_corr_stats_compiled.csv",
                           index=False)
    exp_corr_stats.to_csv(f"{config['main_combo']['table_dir']}/exp_corr_stats_compiled.csv",
                          index=False)
    extended_exp_corr_stats.to_csv(f"{config['main_combo']['table_dir']}/extended_exp_corr_stats_compiled.csv",
                                   index=False)

    endpoint = 'Surrogate'
    exp_corr_stats = exp_corr_stats[exp_corr_stats['endpoint'] == endpoint]
    high_corr_stats = high_corr_stats[high_corr_stats['endpoint'] == endpoint]
    
    fig1 = plot_gini_by_something_boxplot(extended_exp_corr_stats, 'Cancer Type')
    fig1.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.gini_by_cancer_type_boxplot.pdf",
                 bbox_inches='tight')
    fig2 = plot_gini_by_something_boxplot(extended_exp_corr_stats, 'Experimental Class')
    fig2.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.gini_by_experimental_class_boxplot.pdf",
                 bbox_inches='tight')
    fig3 = plot_gini_by_something_boxplot(extended_exp_corr_stats, 'endpoint')
    fig3.savefig(f"{config['main_combo']['fig_dir']}/gini_by_endpoint_boxplot.pdf",
                 bbox_inches='tight')
    
    fig4 = plot_gini_histplot(exp_corr_stats)
    fig4.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.gini_histplot.pdf",
                 bbox_inches='tight')
    fig5 = plot_gini_compare_experimental_and_high(exp_corr_stats, high_corr_stats)
    fig5.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.gini_compare_exp_and_high_kdeplot.pdf",
                 bbox_inches='tight')
    
    fig6 = plot_1mo_responder_percentage(exp_corr_stats, highlight=False)
    fig6.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.1mo_responder_percentage_barplot.pdf",
                 bbox_inches='tight')
    
    fig7 = plot_1mo_responder_percentage(exp_corr_stats, highlight=True)
    fig7.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.1mo_responder_percentage_highlight_barplot.pdf",
                 bbox_inches='tight')
    
    #fig8 = plot_median_benefit_simulated_vs_actual(metadata, exp_corr_stats)
    #fig8.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.median_benefit_simulated_vs_actual_scatterplot.pdf",
    #             bbox_inches='tight')
    
    fig8b = plot_median_benefit_simulated_vs_actual(metadata, exp_corr_stats, highlight=True, endpoint=endpoint)
    fig8b.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.median_benefit_simulated_vs_actual_scatterplot_hightlight.pdf",
                 bbox_inches='tight')
    

if __name__ == '__main__':
    main()


