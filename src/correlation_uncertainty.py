import pandas as pd
import numpy as np
import sys
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from create_input_sheets import create_endpoint_subset_sheet
from utils import load_config

def plot_correlation_distribution(corr_values: np.ndarray) -> plt.Figure:
    """Plot distribution of correlation values

    Args:
        corr_values (np.ndarray): 1D array of correlation values

    Returns:
        plt.Figure: figure of distribution plot
    """
    mean_val = np.mean(corr_values)
    std = np.std(corr_values)
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax = sns.kdeplot(corr_values, ax=ax, 
                     color=sns.color_palette('deep')[0])

    kde_x, kde_y = ax.lines[0].get_data()
    ax.fill_between(kde_x, kde_y, 
                    where=(kde_x > mean_val - std) & (kde_x < mean_val + std),
                    color=sns.color_palette('deep')[0], alpha=0.5)
    
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Density')
    # draw mean and std vertical lines
    ax.axvline(mean_val, color=sns.color_palette('deep')[3], 
               linewidth=1.5, 
               label=f'Mean {mean_val:.2f}')
    #ax.axvline(mean_val - std, color='k', linewidth=1, alpha=0.5,
    #           label=f'SD {std:.2f}')
    #ax.axvline(mean_val + std, color='k', linewidth=1, alpha=0.5)
    ax.set_title(f'{mean_val:.2f} ± {std:.2f} ({len(corr_values)} drug pairs)')
    ax.set_xlim(-0.2, 1)
    ax.set_xticks([0, 0.5, 1])
    #ax.legend()
    return fig


def compile_gini(config: dict) -> pd.DataFrame:
    df = pd.read_excel(config['data_master_sheet'], sheet_name='Sheet1',
                        engine='openpyxl', index_col=0)
    
    df = create_endpoint_subset_sheet(df, 'Surrogate')
    df = df.dropna(subset=['Spearmanr SD'])
    df = df.set_index('key_endpoint')

    estimate_pred_dir = config['main_combo']['table_dir'] + '/predictions' 

    gini_df = pd.DataFrame(index=df.index, 
                           columns=['low', 'high', 'estimate'])

    for key_endpoint in df.index:
        for corr_type in ['low', 'high', 'estimate']:
            corr = df.loc[key_endpoint, 'Spearmanr']
            corr_sd = df.loc[key_endpoint, 'Spearmanr SD']
            if corr_type == 'estimate':
                corr = np.round(corr, 2)
                stats_filename = glob.glob(f'{estimate_pred_dir}/{key_endpoint}/N_500.target_corr_{corr}*.summary_stats.csv')[0]
            elif corr_type == 'low':
                corr = np.round(corr - corr_sd, 2)
                boundary_pred_dir = config['corr_uncertainty_low']['table_dir'] + '/predictions'
                stats_filename = glob.glob(f'{boundary_pred_dir}/{key_endpoint}/N_500.target_corr_{corr}*.summary_stats.csv')[0]
            elif corr_type == 'high':
                boundary_pred_dir = config['corr_uncertainty_high']['table_dir'] + '/predictions'
                corr = np.round(corr + corr_sd, 2)
                stats_filename = glob.glob(f'{boundary_pred_dir}/{key_endpoint}/N_500.target_corr_{corr}*.summary_stats.csv')[0]

            stats = pd.read_csv(stats_filename,
                                index_col=0, header=None).squeeze(axis=1)
            if stats['Used_bestmatch_corr'] == 'True':
                gini_df.loc[key_endpoint, corr_type] = np.nan
            else:
                gini_df.loc[key_endpoint, corr_type] = eval(stats['Gini_coefficient'])

    # drop combos that used bast match corr
    gini_df = gini_df.dropna()
    gini_df.loc[:, 'low_error'] = gini_df['low'] - gini_df['estimate']
    gini_df.loc[:, 'high_error'] = gini_df['estimate'] - gini_df['high']
    gini_df = gini_df.sort_values('estimate')
    return gini_df


def plot_gini_forest(gini_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(1.5, 8))
    gini_df = gini_df.sort_values('estimate')
    ax.errorbar(gini_df['estimate'], gini_df.index, 
                xerr=[gini_df['low_error'], gini_df['high_error']],
                fmt='o',
                markersize=4,
                capsize=3,
                color='k')
    ax.set_xlabel('Gini coefficient')
    ax.set_ylabel('Combination')
    ax.set_xlim(0, 1)
    return fig


def main():
    plt.style.use('env/publication.mplstyle')
    config = load_config()
    gini_df = compile_gini(config)
    errors = gini_df[['low_error', 'high_error']].values.flatten()
    print(f'Error mean: {np.mean(errors):.3f}')
    gini_df.to_csv(f'{config["corr_uncertainty_high"]["table_dir"]}/gini_error_1SD.csv')
    fig = plot_gini_forest(gini_df)
    fig.savefig(f'{config["corr_uncertainty_high"]["fig_dir"]}/gini_forest.pdf')

    # plot correlation distribution example
    corr_df = pd.read_csv(f"{config['cell_line']['data_dir']}/PanCancer_all_pairwise_{config['cell_line']['corr_method']}_correlation.csv", 
                          index_col=0, header=0)
    corr_values = corr_df.values.flatten()
    # remove nans and 1s
    corr_values = corr_values[~np.isnan(corr_values)]
    corr_values = corr_values[corr_values != 1]
    fig = plot_correlation_distribution(corr_values)
    fig.savefig(f'{config["corr_uncertainty_high"]["fig_dir"]}/corr_distribution.pdf')


if __name__ == '__main__':
    main()