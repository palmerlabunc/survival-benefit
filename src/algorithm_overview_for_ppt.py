import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Add the ptdraft folder path to the sys.path list
from survival_benefit.utils import get_weibull_survival_dataframe
from survival_benefit.survival_benefit_class import SurvivalBenefit
from utils import load_config

CONFIG = load_config()
FIGSIZE = (3,2)

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
                solid_capstyle='butt')
    
    ax.set_yticks([0, 50, 100])
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 105)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival (%)')
    return ax

def plot_on_either_side_for_corr(data: pd.DataFrame,
                                 x_neg: str, x_pos: str, y: str,
                                 color_neg: str, color_pos: str,
                                 linewidth: float = 4, y_shift: float = 4,
                                 xlim: float = 24,
                                 ax: plt.Axes = None) -> plt.Axes:
    for i in range(data.shape[0]):
        ax.plot([0, -data.at[i, x_neg]], 
                [data.at[i, y] + y_shift, data.at[i, y] + y_shift], 
                linewidth=linewidth, color=color_neg, 
                solid_capstyle='butt')
        ax.plot([0, data.at[i, x_pos]],
                [data.at[i, y] + y_shift, data.at[i, y] + y_shift],
                linewidth=linewidth, color=color_pos,
                solid_capstyle='butt')
    
    ax.set_xlim(-xlim, xlim)
    ax.set_xticks([-xlim, -xlim/2, 0, xlim, xlim/2])
    ax.set_xticklabels([xlim, xlim/2, 0, xlim, xlim/2])
    ax.set_xlabel('Time (months)')
    ax.set_ylim(0, 105)
    ax.set_ylabel('')
    ax.set_yticks([])
    sns.despine(left=True, bottom=False, right=True, top=True, ax=ax)
    return ax
        
def plot_two_survival_curves(mono: pd.DataFrame, comb: pd.DataFrame, 
                             color_dict: dict, tmax=30) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.lineplot(x='Time', y='Survival', data=mono, ax=ax, 
                color=color_dict['control_arm'], label='A')
    sns.lineplot(x='Time', y='Survival', data=comb, ax=ax, 
                color=color_dict['combination_arm'], label='A+B')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival (%)')
    ax.set_yticks([0, 50, 100])
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, 105)
    
    return fig

def main():
    plt.style.use('env/publication.mplstyle')
    color_dict = CONFIG['colors']
    outdir = CONFIG['example']['fig_dir'] + '/ppt_overview'
    os.makedirs(outdir, exist_ok=True)
    tmax = 30

    mono_u = get_weibull_survival_dataframe(1, 10, 1000)
    comb_u = get_weibull_survival_dataframe(1, 20, 1000)

    sb = SurvivalBenefit(mono_data=mono_u, comb_data=comb_u, 
                         mono_name='A', comb_name='A+B', 
                         n_patients=20, 
                         atrisk=None, save_mode=False)
    sb.set_tmax(tmax)
    sb.compute_benefit_at_corr(0.1, use_bestmatch=True)
    
    # plot 2 survival curves
    fig = plot_two_survival_curves(mono_u, comb_u, color_dict)
    fig.savefig(f'{outdir}/survival_curves.png', bbox_inches='tight')

    # plot A+B
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax = plot_illustrative_survival_bars(sb.benefit_df, 'new_t', 'new_surv', 
                                        color_dict['combination_arm'], 
                                        linewidth=4, y_shift=2, xlim=tmax,
                                        ax=ax)
    sns.lineplot(x='Time', y='Survival', data=comb_u, ax=ax, 
                color=color_dict['combination_arm'], label='A')
    ax.legend().remove()
    fig.savefig(f'{outdir}/A+B.png', bbox_inches='tight')

    # plot A
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax = plot_illustrative_survival_bars(sb.benefit_df, 'Time', 'Survival', 
                                        color_dict['control_arm'], 
                                        linewidth=4, y_shift=2, xlim=tmax,
                                        ax=ax)
    sns.lineplot(x='Time', y='Survival', data=mono_u, ax=ax, 
                color=color_dict['control_arm'], label='A')
    ax.legend().remove()
    fig.savefig(f'{outdir}/A.png', bbox_inches='tight')

    for corr in [0.1, 0.5, 1]:
        sb.compute_benefit_at_corr(corr, use_bestmatch=True)
        # plot sorted by A+B
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax = plot_illustrative_survival_bars(sb.benefit_df, 'new_t', 'new_surv', 
                                            color_dict['added_benefit'], 
                                            linewidth=4, y_shift=2, xlim=tmax,
                                            ax=ax)
        ax = plot_illustrative_survival_bars(sb.benefit_df, 'Time', 'new_surv',
                                            color_dict['control_arm'],
                                            linewidth=4, y_shift=2, xlim=tmax, ax=ax)
        fig.savefig(f'{outdir}/corr_{corr:.1f}_sorted_by_A+B.png', 
                    bbox_inches='tight')
        # plot sorted by A
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax = plot_illustrative_survival_bars(sb.benefit_df, 'new_t', 'Survival', 
                                            color_dict['added_benefit'], 
                                            linewidth=4, y_shift=2, xlim=tmax,
                                            ax=ax)
        ax = plot_illustrative_survival_bars(sb.benefit_df, 'Time', 'Survival',
                                            color_dict['control_arm'],
                                            linewidth=4, y_shift=2, xlim=30, ax=ax)

        fig.savefig(f'{outdir}/corr_{corr:.1f}_sorted_by_A.png', 
                    bbox_inches='tight')
        # plot extracted added benefit
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax = plot_illustrative_survival_bars(sb.benefit_df, 'delta_t', 'Survival', 
                                            color_dict['added_benefit'], 
                                            linewidth=4, y_shift=2, xlim=tmax,
                                            ax=ax)
        fig.savefig(f'{outdir}/corr_{corr:.1f}_added_benefit.png',
                    bbox_inches='tight')
        # plot sorted added benefit
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sorted_deltat = sb.benefit_df.sort_values('delta_t', ascending=False)
        sorted_deltat.loc[:, 'sorted_survival'] = np.arange(0, 100, 100/len(sorted_deltat))
        ax = plot_illustrative_survival_bars(sorted_deltat, 'delta_t', 'sorted_survival', 
                                            color_dict['added_benefit'], 
                                            linewidth=4, y_shift=2, xlim=tmax,
                                            ax=ax)

        fig.savefig(f'{outdir}/corr_{corr:.1f}_sorted_added_benefit.png',
                    bbox_inches='tight')
        
        # plot corr
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax = plot_on_either_side_for_corr(sb.benefit_df, 'Time', 'delta_t', 'Survival', 
                                          color_dict['control_arm'], color_dict['added_benefit'], 
                                          linewidth=4, y_shift=2, xlim=tmax,
                                          ax=ax)

        fig.savefig(f'{outdir}/corr_{corr:.1f}_corr.png',
                    bbox_inches='tight')

if __name__ == '__main__':
    main()

