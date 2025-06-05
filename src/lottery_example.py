import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_config


def fig1(wealth, new_wealth) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(4, 5), 
                             layout='constrained')
    axes = axes.flatten()
    n = len(wealth)

    #### cumulative
    axes[0].plot(range(0, n), np.sort(wealth), 
                color='tab:blue', linewidth=2,
                label='income before lottery')
    axes[0].plot(range(0, n), np.sort(new_wealth), 
                color='tab:orange', linewidth=2,
                label='income after lottery')
    axes[0].set_xlabel('Cumulative distribution')
    axes[0].legend(loc='upper left')

    #### Sort people by original wealth
    sorted_indices = np.argsort(wealth)
    sorted_original_wealth = wealth[sorted_indices]
    sorted_new_wealth = new_wealth[sorted_indices]

    # Create a color gradient based on original wealth
    color_map = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    normalized_original_wealth = (sorted_original_wealth - sorted_original_wealth.min()) / (sorted_original_wealth.max() - sorted_original_wealth.min())
    colors = color_map(normalized_original_wealth)


    axes[1].bar(range(len(sorted_original_wealth)), 
                sorted_new_wealth, color='tab:orange')
    axes[1].bar(range(len(sorted_original_wealth)), 
                sorted_original_wealth, color=colors)

    axes[1].plot(range(0, n), np.sort(wealth), 
                color='tab:blue', linewidth=3)
    axes[1].set_xlabel('Sorted by wealth before lottery')

    #### Sort people by new wealth
    sorted_indices = np.argsort(new_wealth)
    sorted_original_wealth = wealth[sorted_indices]
    sorted_new_wealth = new_wealth[sorted_indices]

    # Create a color gradient based on original wealth
    normalized_original_wealth = (sorted_original_wealth - sorted_original_wealth.min()) / (sorted_original_wealth.max() - sorted_original_wealth.min())
    colors = color_map(normalized_original_wealth)

    # Create the bar plot
    axes[2].bar(range(n), sorted_new_wealth, color='tab:orange', 
                label='lottery winnings')
    axes[2].bar(range(n), sorted_original_wealth, color=colors)

    axes[2].plot(range(0, n), np.sort(new_wealth), 
                color='tab:orange', linewidth=3)
    axes[2].set_xlabel('Sorted by wealth after lottery')

    axes[2].legend(loc='upper left')
    for i in range(3):
        axes[i].set_xlim(-1, n+1)
        axes[i].set_xticks([])
        axes[i].set_xticklabels('')
        axes[i].set_yticks([0, 50000, 100000])
        axes[i].set_yticklabels(['0', '50k', '100k'])

    fig.supylabel('Wealth ($)') 

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=color_map), ax=axes[2],
                        orientation='horizontal', shrink=0.2,
                        label='Wealth before lottery')
    cbar.ax.set_xticks([0, 0.5, 1])
    cbar.ax.set_xticklabels(['0', '50k', '100k'])

    return fig


def cumulative_distribution(wealth, new_wealth) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2, 1.5))
    sns.ecdfplot(wealth, color='tab:blue', 
                 label='Wealth before lottery',
                 linewidth=1.5,
                 ax=ax)
    sns.ecdfplot(new_wealth, color='tab:orange',
                 label='Wealth after lottery',
                 linewidth=1.5,
                 ax=ax)
    
    ax.set_xlim(0, 100000)
    ax.set_xlabel('Wealth ($)')
    ax.set_xticks([0, 50000, 100000])
    ax.set_xticklabels(['0', '50k', '100k'])
    ax.set_ylabel('Cumulative probability')
    ax.set_yticks([0, 0.5, 1])

    return fig


def probability_distribution(wealth, new_wealth) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2, 1.5))
    sns.kdeplot(wealth, color='tab:blue', 
                label='Wealth before lottery',
                linewidth=1.5,
                ax=ax)
    sns.kdeplot(new_wealth, color='tab:orange',
                label='Wealth after lottery',
                linewidth=1.5,
                ax=ax)
    
    ax.set_xlim(0, 100000)
    ax.set_xlabel('Wealth ($)')
    ax.set_xticks([0, 50000, 100000])
    ax.set_xticklabels(['0', '50k', '100k'])
    ax.set_ylabel('Probability')
    
    return fig
    

def main():
    config = load_config()

    plt.style.use('env/publication.mplstyle')

    # generate random normal data
    n = 200
    rng = np.random.default_rng(seed=2024)
    wealth = rng.normal(loc=50000, scale=20000, size=n)
    wealth[wealth < 0] = 0

    # random 25% wins a $20,000 lottery winning
    lottery = rng.choice([0, 20000], size=n, p=[0.75, 0.25])
    new_wealth = wealth + lottery

    fig_dir = config['example']['fig_dir']
    fig = fig1(wealth, new_wealth)
    fig.savefig(f'{fig_dir}/lottery_example.pdf', 
                bbox_inches='tight')

    fig = cumulative_distribution(wealth, new_wealth)
    fig.savefig(f'{fig_dir}/lottery_example_cumulative.pdf', 
                bbox_inches='tight')

    fig = probability_distribution(wealth, new_wealth)
    fig.savefig(f'{fig_dir}/lottery_example_probability.pdf', 
                bbox_inches='tight')



if __name__ == '__main__':
    main()