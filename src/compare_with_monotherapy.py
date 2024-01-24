import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import numpy.typing as npt
from scipy.stats import pearsonr, kstest, wilcoxon
from sklearn.metrics import mean_squared_error
from survival_benefit.survival_data_class import SurvivalData
from survival_benefit.survival_benefit_class import SurvivalBenefit
from survival_benefit.utils import interpolate
import warnings
from utils import load_config, set_figure_size_dim, get_xticks

warnings.filterwarnings("ignore")

def get_valid_deltat(dirname: str, corr: float) -> np.ndarray[float]:
    """Get valid delta_t values for a given correlation parameter value.

    Args:
        dirname (str): directory where the SurvivalBenefit analysis results are stored
        corr (float): correlation parameter value

    Returns:
        np.ndarray[np.float64]: sorted delta_t values (descending order) in valid region
    """
    try:
        benefit_filename = glob.glob(f'{dirname}/N_500.target_corr_{corr:.2f}*.table.csv')[0]
    except IndexError:
        print(dirname)
    stats_filename = glob.glob(f'{dirname}/N_500.target_corr_{corr:.2f}*.summary_stats.csv')[0]
    benefit_df = pd.read_csv(benefit_filename, index_col=0)
    stats = pd.read_csv(stats_filename, index_col=0).squeeze()
    deconv_tmax = eval(stats['Max_time'])

    valid_benefit_df = benefit_df[benefit_df['valid']]
    valid_delta_t = valid_benefit_df['delta_t']
    # enforce tmax
    valid_delta_t[valid_delta_t > deconv_tmax] = deconv_tmax
    
    return valid_benefit_df['delta_t'].sort_values(ascending=False).values


def interpolated_survival(times, survival=None, 
                          tmax: float = None, n=1000) -> np.ndarray[float]:
    """Interpolate survival values at given times.

    Args:
        times (_type_): Time values
        survival (_type_, optional): Survival values at each times. Defaults to None.
        tmax (float, optional): _description_. Defaults to None.
        n (int, optional): _description_. Defaults to 1000.

    Returns:
        np.ndarray[np.float64]: Interpolated survival values
    """
    times = np.sort(times)[::-1] # sort in descending order

    if tmax is not None:
        times[times > tmax] = tmax
    else:
        tmax = times.max()

    if survival is None:
        survival = np.linspace(0, 100, len(times))

    _, uniq_idx = np.unique(times, return_index=True)
    times = times[uniq_idx]
    survival = survival[uniq_idx]

    df = pd.DataFrame({'Time': times, 'Survival': survival})
    f = interpolate(df, x='Time', y='Survival')
    new_times = np.linspace(0, tmax, n)
    return f(new_times)


def make_label(name: str) -> str:
    """Format file name to '{Cancer type} {Drugs}\n{Author} et al. {Year}' for figure label.

    Args:
        name (str): input file prefix in the format of '{Cancer}_{Drug}_{Author}{Year}_PFS'

    Returns:
        str: formatted label
    """

    tokens = name.split('_')
    cancer = tokens[0]
    drugA, drugB = tokens[1].split('-')
    author, year = tokens[2][:-4], tokens[2][-4:]

    dictionary = {"Atezolizumab" :"Atezo.", 
                  "Pembrolizumab":"Pembro.",
                  "Dexamethasone": "Dex.",
                  "Lenalidomide:": "Lena.",
                  "Bevacizumab": "Bev.",
                  "Daratumumab": "Dara.",
                  "Abemaciclib": "Abema.",
                  "Fluorouracil": "5-FU",
                  "Cetuximab": "Cetux.",
                  "Leucovorin": "LV"} 
    for key in dictionary.keys():
        drugA = drugA.replace(key, dictionary[key])
        drugB = drugB.replace(key, dictionary[key])


    return f"{cancer} {drugA}+{drugB}\n({author} et al. {year})"


def plot_compare_monotherapy_added_benefit_one_combo(ax: plt.Axes, monotherapy: pd.DataFrame,
                                                     color_dict: dict,
                                                     target_deltat: np.ndarray[float] = None,
                                                     high_deltat: np.ndarray[float] = None) -> plt.Axes:
    """Plot added benefit survival curves for a given combination on a given Axes.

    Args:
        ax (plt.Axes): Axes to plot on
        monotherapy (pd.DataFrame): monotherapy survival data
        color_dict (dict): color dictionary
        target_deltat (np.ndarray[np.float64], optional): delta_t values for experimental correlation. Defaults to None.
        high_deltat (np.ndarray[np.float64], optional): delta_t values for high correlation. Defaults to None.
    """

    ax.plot('Time', 'Survival', data=monotherapy, 
            linewidth=2, label='Monotherapy', color='k')
    
    if target_deltat is not None:
        ax.plot(target_deltat, 
                np.linspace(0, 100, target_deltat.shape[0]),
                color=color_dict['inference'], 
                label='Inference',
                linewidth=1.2)
    if high_deltat is not None:
        ax.plot(high_deltat, 
                np.linspace(0, 100, high_deltat.shape[0]),
                color=color_dict['visual_appearance'], 
                label='Visual Appearance',
                linewidth=1.2)
    
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks(get_xticks(tmax, metric='months'))
    ax.set_xlabel('Added benefit (months)')
    ax.set_ylabel('Patients (%)')
    
    return ax


def plot_compare_monotherapy_added_benefit_each_combo(input_sheet: pd.DataFrame, 
                                                      data_dir: str, pred_dir: str,
                                                      color_dict: dict) -> plt.Figure:
    """Plot added benefit survival curves for each combination in the input sheet.

    Args:
        input_sheet (pd.DataFrame): input sheet containing combination, control, and correlation
        data_dir (str): directory where all survival data are stored
        pred_dir (str): directory where the SurvivalBenefit analysis results are stored
        color_dict (dict): color dictionary

    Returns:
        plt.Figure: figure containing all plots
    """
    indf = input_sheet
    n = 500

    n_combos = indf.shape[0]
    fig, axes = set_figure_size_dim(n_combos, ax_width=1.5, ax_height=1.5, max_cols=5)
    ax_idx = 0

    for i in indf.index:
        ax = axes[ax_idx]
        ax_idx += 1

        name_exp = indf.at[i, 'Experimental']
        name_ab = indf.at[i, 'Combination']
        corr = indf.at[i, 'Corr']  # experimental spearman correlation value
        
        scan_exp = indf.at[i, 'Experimental First Scan Time']
        
        df_exp = pd.read_csv(f'{data_dir}/{name_exp}.csv',
                            header=0, index_col=False)
        exp = SurvivalData(name_exp, df_exp, n).processed_data

        # subtract scan time
        if scan_exp == 9999:
            scan_exp = 1
            ax.set_title("unknown scan time")
        exp.loc[:, 'Time'] = exp['Time'] - scan_exp
        exp.loc[exp['Time'] < 0, 'Time'] = 0

        dirname = f'{pred_dir}/{name_ab}'
        target_deltat = get_valid_deltat(dirname, corr)
        high_deltat = get_valid_deltat(dirname, 1)

        assess_tmax = np.min([target_deltat.max(), exp['Time'].max()])

        # plot
        ax = plot_compare_monotherapy_added_benefit_one_combo(ax, exp, color_dict, 
                                                              target_deltat=target_deltat, 
                                                              high_deltat=high_deltat)

        
        ax.set_xlim(0, assess_tmax)
        ax.set_xticks(get_xticks(assess_tmax, metric='months'))
        if i == 0:
            ax.legend()
        ax.set_title(make_label(name_ab))
        
    return fig


def compare_monotherapy_added_benefit(input_sheet: pd.DataFrame, data_dir: str, pred_dir: str) -> pd.DataFrame:
    """Compare added benefit survival curves between experimental correlation and high corrrelation 
    for each combination.

    Args:
        input_sheet (pd.DataFrame): _description_
        data_dir (str): directory where all survival data are stored
        pred_dir (str): directory where the SurvivalBenefit analysis results are stored

    Returns:
        pd.DataFrame: rmse and gini values for each combination
    """
    indf = input_sheet
    n = 500
    compare_df = pd.DataFrame(index=indf.index, 
                        columns=['Combination', 'experimental_corr_rmse', 'high_corr_rmse',
                                 'monotherapy_gini', 'experimental_corr_gini', 'high_corr_gini'])
    for i in indf.index:

        name_exp = indf.at[i, 'Experimental']
        name_ab = indf.at[i, 'Combination']
        corr = indf.at[i, 'Corr']  # experimental spearman correlation value
        
        scan_exp = indf.at[i, 'Experimental First Scan Time']
        
        df_exp = pd.read_csv(f'{data_dir}/{name_exp}.csv',
                            header=0, index_col=False)
        exp = SurvivalData(name_exp, df_exp, n).processed_data

        # subtract scan time
        if scan_exp == 9999:
            scan_exp = 1
        exp.loc[:, 'Time'] = exp['Time'] - scan_exp
        exp.loc[exp['Time'] < 0, 'Time'] = 0

        dirname = f'{pred_dir}/{name_ab}'
        target_deltat = get_valid_deltat(dirname, corr)
        high_deltat = get_valid_deltat(dirname, 1)

        assess_tmax = np.min([target_deltat.max(), exp['Time'].max()])

        # calculate Mean Squared Error
        target_surv = interpolated_survival(target_deltat, tmax=assess_tmax)
        high_surv = interpolated_survival(high_deltat, tmax=assess_tmax)
        exp_surv = interpolated_survival(exp['Time'].values, survival=exp['Survival'].values, 
                                         tmax=assess_tmax)
        
        target_rmse = mean_squared_error(exp_surv, target_surv, squared=False)
        high_rmse = mean_squared_error(exp_surv, high_surv, squared=False)

        compare_df.at[i, 'Combination'] = name_ab
        compare_df.at[i, 'experimental_corr_rmse'] = target_rmse
        compare_df.at[i, 'high_corr_rmse'] = high_rmse
        
        compare_df.at[i, 'experimental_corr_gini'] = SurvivalBenefit.compute_gini(target_deltat, assess_tmax)
        compare_df.at[i, 'high_corr_gini'] = SurvivalBenefit.compute_gini(high_deltat, assess_tmax)
        compare_df.at[i, 'monotherapy_gini'] = SurvivalBenefit.compute_gini(exp['Time'].values, assess_tmax)

    return compare_df


def plot_rmse_lineplot(data: pd.DataFrame) -> plt.Figure:
    """

    Args:
        data (pd.DataFrame): compare_df from compare_monotherapy_added_benefit

    Returns:
        plt.figure: figure containing line plot of RMSE values
    """
    stat, p = wilcoxon(data['experimental_corr_rmse'], data['high_corr_rmse'])
    n = data.shape[0]

    fig, ax = plt.subplots(figsize=(1.2, 1.5))
    for i in range(data.shape[0]):
        ax.plot([0, 1], 
                [data.at[i, 'high_corr_rmse'], data.at[i, 'experimental_corr_rmse']], 
                marker='o', color='k', markersize=3, linewidth=0.75)
    ax.set_title(f'Wilcoxon p={p:.1e} (n={n})')
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Visual\nAppearance', 'Inference'])
    ax.set_ylabel('Error (%)')
    return fig


def plot_monotherapy_added_benefit_gini_scatterplot(data: pd.DataFrame, color_dict: dict) -> plt.Figure:
    """_summary_

    Args:
        data (pd.DataFrame): compare_df from compare_monotherapy_added_benefit
        color_dict (dict): color dictionary

    Returns:
        plt.figure: figure containing scatter plot of Gini values
    """
    r, p = pearsonr(data['monotherapy_gini'], data['experimental_corr_gini'])
    n = data.shape[0]

    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    
    sns.scatterplot(x='monotherapy_gini', y='experimental_corr_gini', 
                    data=data,
                    color=color_dict['inference'],
                    size=3,
                    ax=ax)
    ax.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=0.75)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_title(f'Pearsonr={r:.2f}, p={p:.1e} (n={n})')
    ax.set_xlabel('Monotherapy Gini coefficient')
    ax.set_ylabel('Inference Gini coefficient')
    ax.get_legend().remove()
    return fig


def main():
    plt.style.use('env/publication.mplstyle')
    color_dict = load_config()['colors']
    
    config = load_config()['single_agent']
    sheet = config['metadata_sheet']
    data_dir = config['data_dir']
    pred_dir = f"{config['table_dir']}/predictions"
    fig_dir = config['fig_dir']

    indf = pd.read_csv(sheet, header=0, index_col=None)
    
    compare_df = compare_monotherapy_added_benefit(indf, data_dir, pred_dir)
    compare_df.to_csv(f'{config["table_dir"]}/compare_monotherapy_added_benefit.csv', index=False)

    indiv_combo_plot = plot_compare_monotherapy_added_benefit_each_combo(indf, data_dir, pred_dir, color_dict)
    indiv_combo_plot.savefig(f'{fig_dir}/compare_monotherapy_added_benefit_each_combo.pdf',
                             bbox_inches='tight')
    
    example_indf = indf[indf['Combination'] == config['example_combo']]
    example_combo_plot = plot_compare_monotherapy_added_benefit_each_combo(example_indf, data_dir, pred_dir, color_dict)
    example_combo_plot.savefig(f'{fig_dir}/compare_monotherapy_added_benefit_example_combo.pdf',
                               bbox_inches='tight')

    rmse_plot = plot_rmse_lineplot(compare_df)
    rmse_plot.savefig(f'{fig_dir}/compare_monotherapy_added_benefit_rmse.pdf',
                      bbox_inches='tight')
    
    gini_scatterplot = plot_monotherapy_added_benefit_gini_scatterplot(compare_df, color_dict)
    gini_scatterplot.savefig(f'{fig_dir}/compare_monotherapy_added_benefit_gini.pdf',
                             bbox_inches='tight')


if __name__ == '__main__':
    main()