import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
import warnings
import os
import glob
from utils import load_config


def therapy_modality_counts(data_master_df: pd.DataFrame) -> pd.Series:
    df = data_master_df.drop_duplicates(subset=['Cancer Type', 'Experimental', 'Control',
                                                'Trial ID', 'Trial Name'])
    return (df['Experimental Class'] + " + " + df['Control Class']).value_counts(ascending=True)


def plot_therapy_modality_counts(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax = data.plot.barh(ax=ax, color=sns.color_palette('deep')[0])
    ax.set_xlabel('Count')
    ax.set_xticks(range(0, 20, 5))
    return fig


def cancer_type_counts(data_master_df: pd.DataFrame, simplified=False) -> pd.Series:
    df = data_master_df.drop_duplicates(subset=['Cancer Type', 'Experimental', 'Control',
                                                'Trial ID', 'Trial Name'])
    if not simplified:
        count_df = df['Cancer Type'].value_counts(ascending=True)
        return count_df
    
    simplified_cancer_types = {'Mesothelioma': 'Lung',
                               'Gastric': 'Gastrointestinal',
                               'Colorectal': 'Gastrointestinal',
                               'Pancreatic': 'Gastrointestinal',
                               'Ovarian': "Reproductive\nsystem",
                               'Cervical': "Reproductive\nsystem",
                               'Prostate': "Reproductive\nsystem",
                               'HeadNeck': 'Other',
                               'Renal': 'Other'}
    df.loc[:, 'Simplified Cancer Type'] = df['Cancer Type'].replace(simplified_cancer_types)
    count_df = df['Simplified Cancer Type'].value_counts(ascending=True)
    return count_df


def plot_cancer_type_counts(data: pd.DataFrame, stacked=False) -> plt.Figure:
    if not stacked:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax = data.plot.barh(ax=ax, color=sns.color_palette('deep')[0])
        ax.set_xlabel('Count')
        ax.set_xticks(range(0, 20, 5))
        return fig

    fig, ax = plt.subplots(figsize=(0.8, 4))

    category_names = data.index
    category_colors = sns.color_palette('deep', len(category_names))
    data_cum = data.cumsum()
    labels = ['Cancer Types']
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        height = data.iat[i]
        starts = data_cum[i] - height
        rects = ax.bar(labels, height, bottom=starts, width=0.5,
                       label=colname, color=color)
        plt.figtext(1, (starts + height / 2) / data.sum(), colname, 
                    ha='left', va='top', color='black',
                    transform=ax.transAxes)

        r, g, b, _ = to_rgba(color)
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.set_ylabel('Number of combination therapies')
    
    # turn off x axis
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    return fig


def correlation_distribution(data_master_df: pd.DataFrame) -> pd.DataFrame:
    df = data_master_df.drop_duplicates(subset=['Cancer Type', 'Experimental', 'Control',
                                                'Trial ID', 'Trial Name'])
    return df[['Cancer Type', 'Experimental', 'Control',
               'Measured Combo for Correlation', 'Correlation Source Data', 'Pearsonr']]


def plot_correlation_distribution(data: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(data['Pearsonr'], binwidth=0.1, kde=True,
                 color=sns.color_palette('deep')[0], ax=ax)
    ax.set_xlim(0, 1)
    return fig


def main():
    plt.style.use('env/publication.mplstyle')
    config = load_config()

    data_master_df = pd.read_excel(
        config['data_master_sheet'], engine='openpyxl')
    data_master_df = data_master_df.dropna(subset=['Key', 'Indication'])

    # plot therapy modality counts
    therapy_modality_counts_df = therapy_modality_counts(data_master_df)
    therapy_modality_counts_df.to_csv(
        f'{config["main_combo"]["table_dir"]}/therapy_modality_counts_barplot.source_data.csv')
    fig1 = plot_therapy_modality_counts(therapy_modality_counts_df)
    fig1.savefig(f'{config["main_combo"]["fig_dir"]}/therapy_modality_counts_barplot.pdf',
                 bbox_inches='tight')

    # plot cancer type counts
    cancer_type_counts_df = cancer_type_counts(data_master_df)
    cancer_type_counts_df.to_csv(
        f'{config["main_combo"]["table_dir"]}/cancer_type_counts_barplot.source_data.csv')
    fig2 = plot_cancer_type_counts(cancer_type_counts_df)
    fig2.savefig(f'{config["main_combo"]["fig_dir"]}/cancer_type_counts_barplot.pdf',
                 bbox_inches='tight')
    
    simple_cancer_type_counts_df = cancer_type_counts(data_master_df, simplified=True)
    simple_cancer_type_counts_df.to_csv(
        f'{config["main_combo"]["table_dir"]}/simple_cancer_type_counts_barplot.source_data.csv')
    fig2 = plot_cancer_type_counts(simple_cancer_type_counts_df, stacked=True)
    fig2.savefig(f'{config["main_combo"]["fig_dir"]}/simple_cancer_type_counts_barplot.pdf',
                 bbox_inches='tight')

    # plot correlation distribution
    correlation_distribution_df = correlation_distribution(data_master_df)
    correlation_distribution_df.to_csv(
        f'{config["main_combo"]["table_dir"]}/correlation_distribution_histplot.source_data.csv')
    fig3 = plot_correlation_distribution(correlation_distribution_df)
    fig3.savefig(f'{config["main_combo"]["fig_dir"]}/correlation_distribution_histplot.pdf',
                 bbox_inches='tight')


if __name__ == '__main__':
    main()
