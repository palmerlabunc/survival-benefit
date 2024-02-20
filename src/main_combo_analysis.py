import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import glob
from survival_benefit.survival_benefit_class import SurvivalBenefit
from utils import load_config

config = load_config()
COLOR_DICT = config['colors']

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

    fig, ax = plt.subplots(figsize=(3, 1.5))
    if highlight:
        stat_df.plot.bar(y=col, color=color_dict, ax=ax)
        ax.axhline(50, color='black', linestyle='--')
    else:
        stat_df.plot.bar(y=col, ax=ax, color=COLOR_DICT['inference'])

    ax.set_xticklabels('')
    ax.set_xticks([])
    ax.set_ylabel('Patients benefitting\nat least 1 month (%)')
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel('88 Combination therapies')
def plot_1mo_responder_percentage_inferred_apparent(high_corr_stats: pd.DataFrame, 
                                                    exp_corr_stats: pd.DataFrame) -> plt.Figure:
    col = 'Percent_patients_benefitting_1mo_from_valid_highbound'
    exp_stat_df = exp_corr_stats.sort_values(col)
    high_stat_df = high_corr_stats.reindex(exp_stat_df.index)
    fig, ax = plt.subplots(figsize=(3, 1.5))

    high_stat_df.reset_index().plot.bar(y=col, ax=ax, 
                                        color=COLOR_DICT['visual_appearance'])
    exp_stat_df.reset_index().plot.bar(y=col, ax=ax, 
                                    color=COLOR_DICT['inference'])

    ax.set_xticklabels('')
    ax.set_xticks([])
    ax.set_ylabel('Patients benefitting\nat least 1 month (%)')
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel(f'{exp_stat_df.shape[0]} Combination therapies')
    ax.set_ylim(0, 100)
    ax.get_legend().remove()
    return fig


def compare_OS_surrogate_gini(compiled_stats: pd.DataFrame) -> pd.DataFrame:
    df = compiled_stats.pivot_table(index='Key', 
                                    columns='endpoint', 
                                    values='Gini_coefficient')
    df = df.dropna()
    return df


def plot_OS_surrogate_gini(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    sns.scatterplot(data=data, x='OS', y='Surrogate',
                    s=10)
    r, p = pearsonr(data['OS'], data['Surrogate'])
    ax.set_title(f'Pearson r={r:.2f}, p={p:.2f}, n={data.shape[0]}')
    ax.set_xlim(0.2, 1)
    ax.set_ylim(0.2, 1)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.plot([0.2, 1], [0.2, 1], 'k--')
    ax.set_xlabel('OS Gini coefficient')
    ax.set_ylabel('Surrogate Gini coefficient')
    return fig


def plot_median_benefit_simulated_vs_actual(metadata: pd.DataFrame, 
                                            stats: pd.DataFrame,
                                            config: dict,
                                            highlight=False, 
                                            endpoint: str = None) -> tuple[pd.DataFrame, plt.Figure]:
    """_summary_

    Args:
        metadata (pd.DataFrame): _description_
        stats (pd.DataFrame): _description_
        config (dict): _description_
        highlight (bool, optional): Highlight combinations with > 3 months more benefit. Defaults to False.
        endpoint (str, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: source data
        plt.Figure: plotted figure
    """
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

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    if not highlight:
        sns.scatterplot(y='Median_benefit_highbound', 
                        x='actual',
                        size=3,
                        data=merged, 
                        ax=ax, color=COLOR_DICT['inference'])
    else:
        sns.scatterplot(y='Median_benefit_highbound', x='actual',
                        hue='3mo_more_benefit',
                        hue_order=[False, True],
                        size=3,
                        data=merged, 
                        ax=ax, 
                        palette=[sns.color_palette('deep')[0], 'orange'])
    
    # legend box outside of the plot above the right corner
    #ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
    ax.get_legend().remove()
    ax.plot([0, 27], [0, 27], color='black', 
            linestyle='--', linewidth=1)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 27)
    ax.set_xticks([0, 10, 20])
    ax.set_yticks([0, 10, 20])
    ax.set_ylabel('Median months gained\namong patients with benefit')
    ax.set_xlabel('Median months gained\nreported in trial')
    return merged, fig


def plot_gini_histplot(compiled_stats: pd.DataFrame) -> plt.figure:
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.histplot(x=compiled_stats['Gini_coefficient'], binwidth=0.1,
                 binrange=(0, 1),
                 color=sns.color_palette('deep')[4],
                 ax=ax)
    ax.set_xlabel('Gini coefficient')
    ax.set_xlim(0, 1)

    return fig


def plot_gini_compare_experimental_and_high(exp_corr_stats: pd.DataFrame,
                                            high_corr_stats: pd.DataFrame) -> plt.Figure:
    
    exp_corr_stats.loc[:, 'type'] = 'Inference'
    high_corr_stats.loc[:, 'type'] = 'Visual Appearance'

    merged = pd.concat([exp_corr_stats[['combo_name', 'Gini_coefficient', 'type']], 
                        high_corr_stats[['combo_name', 'Gini_coefficient', 'type']]], 
                        axis=0)
    
    fig, axes = plt.subplots(2, 1, figsize=(4, 1.5),
                            gridspec_kw={'height_ratios': [1, 0.2]},
                            sharex=True)

    # kdeplot
    sns.kdeplot(x='Gini_coefficient', hue='type', hue_order=['Visual Appearance', 'Inference'],
                palette=[COLOR_DICT['visual_appearance'], COLOR_DICT['inference']],
                fill=True, alpha=0.5, linewidth=0,
                data=merged, ax=axes[0])

    # rugplot
    sns.scatterplot(x='Gini_coefficient', y='type', hue='type',
                    hue_order=['Visual Appearance', 'Inference'],
                    palette=[COLOR_DICT['visual_appearance'], COLOR_DICT['inference']],
                    data=merged, marker='|', s=20, alpha=0.7, linewidth=1,
                    ax=axes[1])
    # paired t-test
    t, p = ttest_rel(exp_corr_stats.sort_index()['Gini_coefficient'],
                     high_corr_stats.sort_index()['Gini_coefficient'])
    # legend outside of plot on ther right side
    # remove border and legend title
    axes[0].set_title(f'paired t p={p:.1e}')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc=2)
    axes[0].get_legend().set_title('')
    axes[1].set_ylim(-0.5, 1.5)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel('Gini coefficient')
    axes[1].legend().remove()
    axes[1].set_yticklabels('')
    axes[1].set_ylabel('')
    axes[1].set_yticks([])
    
    return fig

def plot_gini_by_something_boxplot(compiled_stats: pd.DataFrame, by: str, n_min=5) -> plt.Figure:
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


def plot_gini_vs_HR(data: pd.DataFrame, endpoint: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    sns.scatterplot(x=f'{endpoint} HR', y='Gini_coefficient',
                    hue='Experimental Class',
                    palette=drug_class_color_dict,
                    s=10,
                    data=data, ax=ax)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xscale('log', base=2)
    x_major = [0.125, 0.25, 0.5, 1]
    ax.xaxis.set_major_locator(plticker.FixedLocator(x_major))
    ax.xaxis.set_major_formatter(plticker.FixedFormatter(x_major))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.xaxis.set_minor_formatter(plticker.NullFormatter())
    ax.set_xlim(0.1, 1.5)
    ax.set_yticks([0, 0.5, 1])

    ax.set_xlabel('Hazard Ratio')
    ax.set_ylabel('Gini coefficient')

    r, p = spearmanr(data['Gini_coefficient'], data[f'{endpoint} HR'])
    ax.set_title(f'Spearmanr={r:.2f}, p={p:.1e}, n={data.shape[0]}')
    return fig


def run_example_survival_benefit(input_df: pd.DataFrame, data_dir: str, pred_dir: str):
    n = 500
    for _, row in input_df.iterrows():
        control_prefix = row['Control']
        comb_prefix = row['Combination']
        corr = row['Corr']

        survival_benefit = SurvivalBenefit(mono_name=f'{data_dir}/{control_prefix}', 
                                            comb_name=f'{data_dir}/{comb_prefix}',
                                            n_patients=n, outdir=pred_dir, 
                                            figsize=(1, 0.8), fig_format='pdf')

        survival_benefit.compute_benefit_at_corr(corr, use_bestmatch=True)
        survival_benefit.plot_compute_benefit_sanity_check(save=True)
        plt.close()
        survival_benefit.plot_t_delta_t_corr(save=True)
        plt.close()
        fig, ax = survival_benefit.plot_benefit_distribution(save=True, simple=True)
        plt.close()
        survival_benefit.save_summary_stats()
        survival_benefit.save_benefit_df()


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

    for endpoint in ['OS', 'Surrogate']:
        exp_corr_stats_endpoint = exp_corr_stats[exp_corr_stats['endpoint'] == endpoint]
        high_corr_stats_endpoint = high_corr_stats[high_corr_stats['endpoint'] == endpoint]
        
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
        
        fig5 = plot_gini_compare_experimental_and_high(exp_corr_stats_endpoint, 
                                                       high_corr_stats_endpoint)
        fig5.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.gini_compare_exp_and_high_kdeplot.pdf",
                    bbox_inches='tight')
        
        fig6 = plot_1mo_responder_percentage(exp_corr_stats_endpoint, highlight=False)
        fig6.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.1mo_responder_percentage_barplot.pdf",
                    bbox_inches='tight')
        
        # plot both inferred and apparent
        fig6 = plot_1mo_responder_percentage_inferred_apparent(high_corr_stats_endpoint,
                                                               exp_corr_stats_endpoint)
        
        fig6.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.1mo_responder_percentage_both_barplot.pdf",
            bbox_inches='tight')
        
        fig7 = plot_1mo_responder_percentage(exp_corr_stats_endpoint, highlight=True)
        fig7.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.1mo_responder_percentage_highlight_barplot.pdf",
                    bbox_inches='tight')
        
        fig8dat, fig8 = plot_median_benefit_simulated_vs_actual(metadata, exp_corr_stats_endpoint)
        fig8.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.median_benefit_simulated_vs_actual_scatterplot.pdf",
                    bbox_inches='tight')
        
        fig8dat, fig8b = plot_median_benefit_simulated_vs_actual(metadata, exp_corr_stats_endpoint, 
                                                        highlight=True, endpoint=endpoint)
        fig8b.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.median_benefit_simulated_vs_actual_scatterplot_hightlight.pdf",
                    bbox_inches='tight')
        fig8dat.to_csv(f"{config['main_combo']['table_dir']}/{endpoint}.median_benefit_simulated_vs_actual_scatterplot_hightlight.source_data.csv")
    
        fig9 = plot_gini_vs_HR(extended_exp_corr_stats_endpoint, endpoint)
        fig9.savefig(f"{config['main_combo']['fig_dir']}/{endpoint}.gini_vs_HR_scatterplot.pdf",
                    bbox_inches='tight')

    # 8: Breast_Ixabepilone-Capecitabine_Thomas2007_PFS
    # 16: Breast_Ribociclib-Letrozole_Hortobagyi2018_PFS
    # 63: Lung_Atezolizumab-Carboplatin+Etoposide_Horn2018_PFS
    example_idx = [8, 16, 63]
    input_df = pd.read_csv(config['main_combo']['metadata_sheet'], index_col=None, header=0)
    input_df = input_df.iloc[example_idx, :]
    data_dir = config['main_combo']['data_dir']
    pred_dir = config['main_combo']['fig_dir']
    run_example_survival_benefit(input_df, data_dir, pred_dir)


if __name__ == '__main__':
    main()


