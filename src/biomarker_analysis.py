import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import os
import glob
import numpy as np
from lifelines import CoxPHFitter
from utils import load_config, get_cox_results
from create_input_sheets import create_endpoint_subset_sheet

def ipd_from_benefit_df2(benefit_df: pd.DataFrame, tmax: float, benefiter=True, arm='monotherapy') -> pd.DataFrame:
    """
    Convert digitized KM curves to individual patient data (IPD) format. Pad the data with censoring times
    (maintain the same proportion as total patients)

    Args:
        benefit_df (pd.DataFrame): The benefit_df output from SurvivalBenefit
        tmax (float): The maximum time of follow-up
        benefiter (bool, optional): Whether to return the benefitting subpopulation. 
            Defaults to True.
        arm (str, optional): The arm to return. 'monotherapy' or 'combination' 
            Defaults to 'monotherapy'.
    
    Returns:
        pd.DataFrame: IPD in the format of Time, Event
    """

    dat = benefit_df.copy()
    event_prop = dat['valid'].sum() / len(dat)

    if benefiter:
        dat = dat[dat['delta_t'] >= 1]
    else:
        dat = dat[dat['delta_t'] < 1]
    
    if arm == 'monotherapy':
        event_times = dat[dat['valid']]['Time'].values
    elif arm == 'combination':
        event_times = dat[dat['valid']]['new_t'].values
    
    event = np.vstack((event_times, np.repeat(1, len(event_times)))).T   
    
    censor_times = np.repeat(tmax, len(event_times)/event_prop * (1 - event_prop))
    censor = np.vstack((censor_times, np.repeat(0, len(censor_times)))).T

    ipd = pd.DataFrame(np.vstack((event, censor)), 
                       columns=['Time', 'Event'])

    censored_idx = ipd[ipd['Time'] >= tmax].index
    ipd.loc[censored_idx, 'Time'] = tmax
    ipd.loc[censored_idx, 'Event'] = 0

    return ipd

def optimal_stratification_analysis(data_df: pd.DataFrame, pred_dir: str) -> pd.DataFrame:
    """Cox PH analysis of optimal stratification.

    Args:
        data_df (pd.DataFrame): dataframe
        pred_dir (str): directory containing SurvivalBenefit prediction outputs

    Returns:
        pd.DataFrame: analysis dataframe
    """    
    dat = data_df.copy()
    for combo in dat.index:
        dirname = os.path.join(pred_dir, combo)
        if not os.path.isdir(dirname):
            continue
        if len(os.listdir(dirname)) == 0:
            continue

        corr = dat.loc[combo, 'Spearmanr']

        benefit_filename = glob.glob(
            f'{dirname}/N_500.target_corr_{corr:.2f}.*.table.csv')[0]

        stats_filename = glob.glob(
            f'{dirname}/N_500.target_corr_{corr:.2f}.*.summary_stats.csv')[0]
        
        stats = pd.read_csv(stats_filename,
                            index_col=0, header=None).squeeze(axis=1)
        tmax = eval(stats['Max_time'])
        benefit_df = pd.read_csv(benefit_filename, 
                                 index_col=0, header=0).squeeze(axis=1)
        
        # selected patients
        for benefiter, label in zip([True, False], ['benefiter', 'non-benefiter']):
            mono_ipd = ipd_from_benefit_df2(benefit_df, tmax, 
                                            benefiter=benefiter, 
                                            arm='monotherapy')
            comb_ipd = ipd_from_benefit_df2(benefit_df, tmax, 
                                            benefiter=benefiter, 
                                            arm='combination')

            p, hr, low, high = get_cox_results(mono_ipd, comb_ipd)
            dat.loc[combo, f'HR_{label}'] = hr
            dat.loc[combo, f'p_{label}'] = p
            dat.loc[combo, f'HR_low_{label}'] = low
            dat.loc[combo, f'HR_high_{label}'] = high

    dat.loc[:, 'better_2x'] = False
    dat.loc[dat['HR']/dat['HR_benefiter'] >= 2, 'better_2x'] = True

    return dat
    
def optimal_stratification_scatterplot(dat: pd.DataFrame, 
                                       subgroup: str = 'benefiter') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.set_palette('deep')
    red = sns.color_palette()[3]
    if subgroup == 'benefiter':
        sns.scatterplot(data=dat, y='HR', x=f'HR_{subgroup}', ax=ax, 
                        hue='better_2x', palette=['#969696', red],
                        size='better_2x', sizes=[10, 12], legend=False)
    
        for combo in dat[dat['better_2x']].index:
            ax.plot([dat.loc[combo, 'HR'], dat.loc[combo, f'HR_{subgroup}']], 
                    [dat.loc[combo, 'HR'], dat.loc[combo, 'HR']], 
                    color=red, alpha=0.5, linewidth=0.3)
            
    else:
        sns.scatterplot(data=dat, y='HR', x=f'HR_{subgroup}', ax=ax,
                        legend=False,
                        c='#969696', s=10)
        
    ax.plot([0, 1], [0, 1], color='black', 
            linestyle='--', linewidth=0.5)

    # convert the axes to log scale
    ax.set_xscale('log', base=2)
    ax.set_xlim(0.1, 1.2)
    ax.set_yscale('log', base=2)
    ax.set_ylim(0.1, 1.2)
    major = [0.125, 0.25, 0.5, 1]
    
    if subgroup == 'benefiter':
        ax.set_xlabel('HRbenefiter')
    else:
        ax.set_xlabel('HRnon-benefiter')
    ax.set_ylabel('HRtotal')

    ax.xaxis.set_major_locator(plticker.FixedLocator(major))
    ax.xaxis.set_major_formatter(plticker.FixedFormatter(major))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.xaxis.set_minor_formatter(plticker.NullFormatter())

    # do the same thing for y axes
    ax.yaxis.set_major_locator(plticker.FixedLocator(major))
    ax.yaxis.set_major_formatter(plticker.FixedFormatter(major))
    ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.yaxis.set_minor_formatter(plticker.NullFormatter())
    
    return fig

def actual_biomarker_vs_optimal_stratification(biomarker_input_df: pd.DataFrame,
                                               pred_dir: str) -> pd.DataFrame:
    """Compare the biomarker analysis with optimal stratification analysis.

    Args:
        biomarker_input_df (pd.DataFrame): biomarker input dataframe
        pred_dir (str): directory containing SurvivalBenefit prediction outputs
    
    Returns:
        pd.DataFrame: analysis dataframe
    
    """

    biomarker_df = biomarker_input_df.copy()
    for idx in biomarker_df.index:
        combo = biomarker_df.loc[idx, 'Combination']
        dirname = os.path.join(pred_dir, combo)
        if not os.path.isdir(dirname):
            continue
        if len(os.listdir(dirname)) == 0:
            continue

        print(dirname)
        corr = biomarker_df.loc[idx, 'Corr']
        benefit_filename = glob.glob(f'{dirname}/N_500.target_corr_{corr:.2f}*.table.csv')[0]
        stats_filename = glob.glob(f'{dirname}/N_500.target_corr_{corr:.2f}*.summary_stats.csv')[0]
        
        stats = pd.read_csv(stats_filename,
                            index_col=0, header=None).squeeze(axis=1)
        tmax = eval(stats['Max_time'])
        benefit_df = pd.read_csv(benefit_filename, index_col=0, header=0)
        
        mono_ipd = ipd_from_benefit_df2(benefit_df, tmax, benefiter=True, 
                                        arm='monotherapy')
        comb_ipd = ipd_from_benefit_df2(benefit_df, tmax, benefiter=True, 
                                        arm='combination')

        p, hr, low, high = get_cox_results(mono_ipd, comb_ipd)
        biomarker_df.loc[idx, 'Optimal_Biomarker_HR'] = hr
        biomarker_df.loc[idx, 'Optimal_Biomarker_p'] = p
        biomarker_df.loc[idx, 'Optimal_Biomarker_HR_95low'] = low
        biomarker_df.loc[idx, 'Optimal_Biomarker_HR_95high'] = high
    
    return biomarker_df


def add_empty_entries_with_adjusted_ci(data, labels, ci_low, ci_high, space_after: int):
    new_data, new_labels, new_ci_low, new_ci_high = [], [], [], []
    for i in range(len(data)):
        new_data.append(data[i])
        new_labels.append(labels[i])
        new_ci_low.append(ci_low[i])
        new_ci_high.append(ci_high[i])
        if (i + 1) % space_after == 0 and i < len(data) - 1:
            new_data.append(None)  # Add a None value for spacing
            new_labels.append('')  # Add an empty label for spacing
            # Do not add None for ci_low and ci_high, skip them instead
    return new_data, new_labels, new_ci_low, new_ci_high


def plot_actual_biomarker_vs_optimal_stratification(biomarker_df: pd.DataFrame) -> plt.figure:
    hr = pd.melt(biomarker_df, 
             id_vars=['Label', 'Biomarker'], 
             value_vars=['ITT_HR', 'Biomarker_HR', 'Optimal_Biomarker_HR'])
    hr_low = pd.melt(biomarker_df, 
                     id_vars=['Label', 'Biomarker'], 
                     value_vars=['ITT_HR_95low', 'Biomarker_HR_95low', 'Optimal_Biomarker_HR_95low'])
    hr_high = pd.melt(biomarker_df, 
                     id_vars=['Label', 'Biomarker'], 
                     value_vars=['ITT_HR_95high', 'Biomarker_HR_95high', 'Optimal_Biomarker_HR_95high'])
    
    hr.loc[:, 'variable'] = pd.Categorical(hr['variable'], 
                                          ["ITT_HR", "Biomarker_HR", "Optimal_Biomarker_HR"])
    hr_low.loc[:, 'variable'] = pd.Categorical(hr_low['variable'], 
                                              ["ITT_HR_95low", "Biomarker_HR_95low", "Optimal_Biomarker_HR_95low"])
    hr_high.loc[:, 'variable'] = pd.Categorical(hr_high['variable'], 
                                               ["ITT_HR_95high", "Biomarker_HR_95high", "Optimal_Biomarker_HR_95high"])

    hr = hr.sort_values(['Label', 'Biomarker', 'variable'])
    hr_low = hr_low.sort_values(['Label', 'Biomarker', 'variable'])
    hr_high = hr_high.sort_values(['Label', 'Biomarker', 'variable'])

    hr_vals = hr['value'].values
    ci_low = hr['value'].values - hr_low['value'].values
    ci_high = hr_high['value'].values - hr['value'].values
    labels = [val for pair in zip(biomarker_df['Label'], biomarker_df['Biomarker'], [f'{i} Ideal Stratification' for i in range(biomarker_df.shape[0])]) for val in pair]

    # Modify the data, labels and confidence intervals with spaces
    spaced_hr_vals, spaced_labels, spaced_ci_low, spaced_ci_high = add_empty_entries_with_adjusted_ci(hr_vals, labels, ci_low, ci_high, 3)

    # Filter out None values for plotting error bars
    filtered_hr_vals = [x for x in spaced_hr_vals if x is not None]
    filtered_ci_low = [x for x in spaced_ci_low if x is not None]
    filtered_ci_high = [x for x in spaced_ci_high if x is not None]
    filtered_ci = np.array([filtered_ci_low, filtered_ci_high])

    # Create a new plot with the modified data
    fig, ax = plt.subplots(figsize=(2, 4))

    # Add error bars with the filtered data
    ax.errorbar(x=filtered_hr_vals,
                y=[i for i, x in enumerate(spaced_hr_vals) if x is not None],  # Y positions for valid data points
                xerr=filtered_ci,
                color='k', capsize=3, linestyle='None',
                linewidth=1, marker="o", markersize=4, mfc="black", mec="black")

    # Set other plot properties as before
    ax.axvline(x=1, linewidth=0.8, linestyle='--', color='red', alpha=0.5)
    ax.invert_yaxis()
    ax.set_xscale('log', base=2)
    x_major = [0.125, 0.25, 0.5, 1]
    ax.xaxis.set_major_locator(plticker.FixedLocator(x_major))
    ax.xaxis.set_major_formatter(plticker.FixedFormatter(x_major))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.xaxis.set_minor_formatter(plticker.NullFormatter())
    ax.set_xlim(0.1, 1.5)

    ax.set_yticks(range(len(spaced_labels)))
    ax.set_yticklabels(spaced_labels)
    ax.set_xlabel('Hazard Ratio')
    return fig


def main():
    plt.style.use('env/publication.mplstyle')
    config = load_config()
    endpoint = 'Surrogate'
    pred_dir = f"{config['main_combo']['table_dir']}/predictions"
    data_master_df = pd.read_excel(config['data_master_sheet'], engine='openpyxl')
    data_master_df = data_master_df.dropna(subset=['Key', 'Indication'])
    data_master_df.set_index('Key', inplace=True)

    pfs_df = create_endpoint_subset_sheet(data_master_df, endpoint)
    pfs_df = pfs_df.set_index('key_endpoint')

    pfs_df = optimal_stratification_analysis(pfs_df, pred_dir)
    pfs_df.to_csv(f"{config['biomarker']['table_dir']}/PFS.optimal_stratification_analysis.csv")
    for subgroup in ['benefiter', 'non-benefiter']:
        fig = optimal_stratification_scatterplot(pfs_df, subgroup=subgroup)
        fig.savefig(f"{config['biomarker']['fig_dir']}/PFS.optimal_stratification_{subgroup}_scatterplot.pdf", 
                    bbox_inches='tight')

    biomarker_df = pd.read_csv(config['biomarker']['metadata_sheet'])
    biomarker_pred_dir = f"{config['biomarker']['table_dir']}/predictions"

    biomarker_df = actual_biomarker_vs_optimal_stratification(biomarker_df, biomarker_pred_dir)
    biomarker_df.to_csv(f"{config['biomarker']['table_dir']}/actual_biomarker_vs_optimal_stratification.csv")
    fig = plot_actual_biomarker_vs_optimal_stratification(biomarker_df)
    fig.savefig(f"{config['biomarker']['fig_dir']}/actual_biomarker_vs_optimal_stratification.pdf", 
                bbox_inches='tight')


if __name__ == '__main__':
    main()
    
