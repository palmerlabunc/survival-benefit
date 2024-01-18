import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Add the ptdraft folder path to the sys.path list
from survival_benefit.utils import generate_weibull, get_weibull_survival_dataframe, interpolate
from survival_benefit.survival_benefit_class import SurvivalBenefit
from utils import load_config

FIGSIZE = (1.5, 1.2)

def plot_illustrative_survival_bars(data: pd.DataFrame,
                                    x: str, y: str,
                                    color: str,
                                    linewidth: float = 4, y_shift: float = 4,
                                    xlim: float = 24,
                                    ax: plt.Axes = None) -> plt.Axes:
    for i in range(data.shape[0]):
        ax.plot([0, data.at[i, x]], 
                [data.at[i, y] + y_shift, data.at[i, y] + y_shift], 
                linewidth=linewidth, color=color, 
                solid_capstyle='butt',)
    
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Patients (%)')
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([0, 12, 24])
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 105)
    return ax


def plot_two_survival_curves(mono: pd.DataFrame, comb: pd.DataFrame, color_dict: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.despine()
    sns.lineplot(x='Time', y='Survival', data=mono, ax=ax, 
                color=color_dict['control_arm'], label='A')
    sns.lineplot(x='Time', y='Survival', data=comb, ax=ax, 
                color=color_dict['combination_arm'], label='A+B')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Patients (%)')
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([0, 12, 24])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 105)
    
    return fig


def plot_sorted_by_A(data: pd.DataFrame, color_dict: dict) -> (plt.Figure, plt.Axes):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax = plot_illustrative_survival_bars(data, 'new_t', 'Survival', 
                                         color_dict['added_benefit'], 
                                         linewidth=3, y_shift=-2, ax=ax)
    ax = plot_illustrative_survival_bars(data, 'Time', 'Survival',
                                         color_dict['control_arm'], 
                                         linewidth=3, y_shift=-2, ax=ax)
    return fig, ax


def plot_sorted_by_AB(data: pd.DataFrame, color_dict: dict) -> (plt.Figure, plt.Axes):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax = plot_illustrative_survival_bars(data, 'new_t', 'new_surv', 
                                         color_dict['added_benefit'], 
                                         linewidth=3, y_shift=-2, ax=ax)
    ax = plot_illustrative_survival_bars(data, 'Time', 'new_surv',
                                         color_dict['control_arm'], 
                                         linewidth=3, y_shift=-2, ax=ax)
    return fig, ax


def main():
    plt.style.use('env/publication.mplstyle')
    config = load_config()
    color_dict = config['colors']
    fig_dir = config['example']['fig_dir']
    
    mono = get_weibull_survival_dataframe(1, 10, 15)
    comb = get_weibull_survival_dataframe(1, 15, 15)

    mono_u = get_weibull_survival_dataframe(1, 10, 1000)
    comb_u = get_weibull_survival_dataframe(1, 15, 1000)

    fig = plot_two_survival_curves(mono_u, comb_u, color_dict)
    fig.savefig(f"{fig_dir}/two_arms.pdf", bbox_inches='tight')

    sb = SurvivalBenefit(mono_u, comb_u, 'A', 'A+B', 15, 
                         atrisk=None, save_mode=False)
    sb.set_tmax(30)
    for corr in [0.3, 1]:
        sb.compute_benefit_at_corr(corr, use_bestmatch=True)
        
        fig, ax = plot_sorted_by_A(sb.benefit_df, color_dict)
        fig.savefig(f"{fig_dir}/sorted_by_A_corr_{corr}.pdf", bbox_inches='tight')
        sns.lineplot(x='Time', y='Survival', data=mono_u, ax=ax, 
                    color=color_dict['control_arm'], label='A')
        fig.savefig(f"{fig_dir}/sorted_by_A_corr_{corr}_with_weibull.pdf", bbox_inches='tight')
        
        fig, ax = plot_sorted_by_AB(sb.benefit_df, color_dict)
        fig.savefig(f"{fig_dir}/sorted_by_AB_corr_{corr}.pdf", bbox_inches='tight')
        sns.lineplot(x='Time', y='Survival', data=comb_u, ax=ax, 
                    color=color_dict['combination_arm'], label='A+B')
        fig.savefig(f"{fig_dir}/sorted_by_AB_corr_{corr}_with_weibull.pdf", bbox_inches='tight')


if __name__ == '__main__':
    main()