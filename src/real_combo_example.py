import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Mapping
from survival_benefit.survival_benefit_class import SurvivalBenefit
from survival_benefit.survival_data_class import SurvivalData
from utils import load_config, get_xticks

FIGSIZE = (1.5, 1.2)

def plot_two_arms(color_dict: Mapping, sb: SurvivalBenefit = None, 
                  data_dir: str = None, 
                  control_prefix: str = None, 
                  combo_prefix: str = None,
                  control_name: str = None,
                  combo_name: str = None
                  ) -> plt.Figure:
    """Provide either the SurvivalBenefit object or the data_dir, control_prefix, and combo_prefix.

    Args:
        color_dict (Mapping): _description_
        sb (SurvivalBenefit, optional): _description_. Defaults to None.
        data_dir (str, optional): _description_. Defaults to None.
        control_prefix (str, optional): _description_. Defaults to None.
        combo_prefix (str, optional): _description_. Defaults to None.

    Returns:
        plt.Figure: _description_
    """
    width, height = FIGSIZE
    width = width * 1.5
    fig, ax = plt.subplots(figsize=(width, height))

    if isinstance(sb, SurvivalBenefit):
        mono_data=sb.mono_survival_data.processed_data
        comb_data=sb.comb_survival_data.processed_data

    else:
        df_mono = pd.read_csv(f'{data_dir}/{control_prefix}.csv', header=0, index_col=False)
        df_mono.columns = ['Time', 'Survival']
        df_comb = pd.read_csv(f'{data_dir}/{combo_prefix}.csv', header=0, index_col=False)
        df_comb.columns = ['Time', 'Survival']

        mono_data = SurvivalData('control_arm', df_mono, 500, weibull_tail=False).processed_data
        comb_data = SurvivalData('combination_arm', df_comb, 500, weibull_tail=False).processed_data

    ax.plot('Time', 'Survival',
            data=mono_data,
            color=color_dict['control_arm'],
            label=control_name)
    ax.plot('Time', 'Survival',
            data=comb_data,
            color=color_dict['combination_arm'],
            label=combo_name)
    
    ax.fill_betweenx(mono_data['Survival'],
                        mono_data['Time'],
                        comb_data['Time'],
                        color=color_dict['added_benefit'],
                        alpha=0.5,
                        label='Added benefit')
        
        
    # legend outside of the plot on top
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), 
              ncol=2, frameon=False)
    if isinstance(sb, SurvivalBenefit):
        ax.set_xlim(0, sb.tmax-1)
    else:
        tmax = min(mono_data['Time'].max(), comb_data['Time'].max())
        ax.set_xlim(0, tmax - 1)
    
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Patients (%)')
    return fig


def plot_experimental_arm(filepath: str, first_scan: float = 0) -> plt.Figure:
    df = pd.read_csv(filepath, header=0, index_col=False)
    df.columns = ['Time', 'Survival']

    surv_dat = SurvivalData('experimental_monotherapy', df, 500)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    data = surv_dat.processed_data.copy()
    data.loc[:, 'Time'] = data['Time'] - first_scan
    ax.plot('Time', 'Survival', 
            data=data,
            color='darkorange')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 6, 12])
    ax.set_yticks([0, 50, 100])
    if first_scan == 0:
        ax.set_xlabel('PFS (months)')
    else:
        ax.set_xlabel('PFS - First Scan (months)')
    ax.set_ylabel('Patients (%)')
    return fig


def main():
    config = load_config()
    color_dict = config['colors']
    data_dir = config['example']['data_dir']
    pred_dir = config['example']['table_dir'] + '/predictions'
    fig_dir = config['example']['fig_dir']

    mono_prefix = config['example']['real_example_control']
    comb_prefix = config['example']['real_example_combo']
    expr_prefix = config['example']['real_example_experimental']

    sb = SurvivalBenefit(mono_name=f'{data_dir}/{mono_prefix}',
                         comb_name=f'{data_dir}/{comb_prefix}',
                         n_patients=500,
                         outdir=pred_dir,
                         fig_format='pdf',
                         figsize=FIGSIZE)
    
    fig = plot_two_arms(color_dict, data_dir=data_dir, 
                        control_prefix=mono_prefix, 
                        combo_prefix=comb_prefix,
                        control_name='Capecitabine',
                        combo_name='Capeicitabine + Lapatinib')
    
    fig.savefig(f'{fig_dir}/{comb_prefix}.two_arms.pdf', bbox_inches='tight')
    
    fig = plot_experimental_arm(f'{data_dir}/{expr_prefix}.csv')
    fig.savefig(f'{fig_dir}/{expr_prefix}.pdf', bbox_inches='tight')

    for corr in [0.25, 0.5, 1]:
        sb.compute_benefit_at_corr(corr, use_bestmatch=True)
        sb.save_benefit_df()
        fig, ax = sb.plot_benefit_distribution(simple=True, save=False)
        gini = sb.stats['Gini_coefficient']
        ax.set_xlabel(f"Benefit from Lapatinib (months)")
        ax.set_xlim(0, 12)
        ax.set_xticks([0, 6, 12])
        ax.set_yticks([0, 50, 100])
        ax.set_title(r'$\rho$ = {0:.2f}'.format(sb.corr_rho_actual))
        ax.text(0.5, 0.6, f'Gini = {gini:.2f}', transform=ax.transAxes)
        fig.savefig(f'{fig_dir}/{comb_prefix}.{sb.info_str}.distribution_simple_absolute.pdf', 
                    bbox_inches='tight')
        
        plt.close()

if __name__ == '__main__':
    main()

