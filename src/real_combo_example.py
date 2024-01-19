import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from survival_benefit.survival_benefit_class import SurvivalBenefit
from survival_benefit.survival_data_class import SurvivalData
from utils import load_config

FIGSIZE = (1.5, 1.2)

def plot_two_arms(sb: SurvivalBenefit, color_dict: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    ax.plot('Time', 'Survival', 
            data=sb.mono_survival_data.processed_data,
            color=color_dict['control_arm'],
            label='FOLFOX4')
    
    ax.plot('Time', 'Survival', 
            data=sb.comb_survival_data.processed_data,
            color=color_dict['combination_arm'],
            label='Bevacizumab + FOLFOX4')
    
    ax.fill_betweenx(sb.mono_survival_data.processed_data['Survival'],
                    sb.mono_survival_data.processed_data['Time'],
                    sb.comb_survival_data.processed_data['Time'],
                    color=color_dict['added_benefit'],
                    alpha=0.5,
                    label='Added benefit')
    
    # legend outside of the plot on the right side
    ax.legend(loc='lower left', frameon=False, bbox_to_anchor=(1.05, 0.0))
    ax.set_xlim(0, sb.tmax-1)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 12, 24])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Patients (%)')
    return fig


def plot_experimental_arm(filepath) -> plt.Figure:
    df = pd.read_csv(filepath, header=0, index_col=False)
    df.columns = ['Time', 'Survival']

    surv_dat = SurvivalData('experimental_monotherapy', df, 500)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    data = surv_dat.processed_data.copy()
    data.loc[:, 'Time'] = data['Time'] - 9/4
    ax.plot('Time', 'Survival', 
            data=data,
            color='darkorange')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel('PFS - First Scan (months)')
    ax.set_ylabel('Patients (%)')
    return fig


def main():
    config = load_config()
    color_dict = config['colors']
    data_dir = config['example']['data_dir']
    pred_dir = config['example']['table_dir'] + '/predictions'
    fig_dir = config['example']['fig_dir']

    mono_name = config['example']['real_example_control']
    comb_name = config['example']['real_example_combo']
    expr_name = config['example']['real_example_experimental']

    sb = SurvivalBenefit(mono_name=f'{data_dir}/{mono_name}',
                         comb_name=f'{data_dir}/{comb_name}',
                         n_patients=500,
                         outdir=pred_dir,
                         fig_format='pdf',
                         figsize=FIGSIZE)
    
    fig = plot_two_arms(sb, color_dict)
    fig.savefig(f'{fig_dir}/{comb_name}.two_arms.pdf', bbox_inches='tight')
    
    fig = plot_experimental_arm(f'{data_dir}/{expr_name}.csv')
    fig.savefig(f'{fig_dir}/{expr_name}.pdf', bbox_inches='tight')

    for corr in [0.24, 0.6, 1]:
        sb.compute_benefit_at_corr(corr, use_bestmatch=True)
        sb.save_benefit_df()
        fig, ax = sb.plot_benefit_distribution(simple=True, save=False)
        ax.set_xticks([0, 6, 12, 18])
        ax.set_yticks([0, 50, 100])
        ax.set_title(r'$\rho$ = %.2f' % sb.corr_rho_actual)
        fig.savefig(f'{fig_dir}/{comb_name}.{sb.info_str}.distribution_simple_absolute.pdf', 
                    bbox_inches='tight')
        
        plt.close()

if __name__ == '__main__':
    main()

