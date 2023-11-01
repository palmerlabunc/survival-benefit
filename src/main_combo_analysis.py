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


def gini_index(values, tmax):
    values = np.sort(values)
    # cap the values at tmax
    values[values > tmax] = tmax
    n = len(values)
    gini = (2 * np.sum(np.multiply(np.arange(1, n+1), values)) /
            (n * np.sum(values))) - (n + 1) / n
    return np.round(gini, 2)

def endpoint_simple(endpoint: str) -> str:
    if endpoint == 'OS':
        return 'OS'
    return 'Surrogate'

def compile_stats(table_dir: str, corr_type: str):
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

        benefit_arr = benefit_df[benefit_df['valid']]['delta_t'].values
        tmax = eval(stats['Max_time'])
        stats.loc['gini_coefficient'] = gini_index(benefit_arr, tmax)
        
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
    
    return stat_df.T


def plot_gini_curve(benefit_df: pd.DataFrame) -> plt.figure:
    df = benefit_df[benefit_df['valid']]
    cumsum = df['delta_t'].sort_values().cumsum().values
    norm_cumsum = 100 * cumsum / cumsum[-1]
    index = np.linspace(0, 100, len(norm_cumsum))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(index, norm_cumsum)
    ax.set_xlabel('Patients (%)')
    ax.set_ylabel('Cumulative benefit (%)')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    ax.plot([0, 100], [0, 100], linestyle='--', color='black')
    return fig


def plot_1mo_responder_percentage(data: pd.DataFrame) -> plt.figure:
    sns.set_palette('deep')
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(data['Percentage of patients benefitting 1mo'],
                 binwidth=10, kde=True, ax=ax)
    ax.set_xlim(0, 1)
    return fig


def main():
    config = load_config()
    high_corr_stats = compile_stats(config['main_combo']['table_dir'], 'high')
    exp_corr_stats = compile_stats(config['main_combo']['table_dir'], 'exp')

    high_corr_stats.to_csv(f"{config['main_combo']['table_dir']}/high_corr_stats_compiled.csv",
                           index=False)
    exp_corr_stats.to_csv(f"{config['main_combo']['table_dir']}/exp_corr_stats_compiled.csv",
                          index=False)
    

if __name__ == '__main__':
    main()


