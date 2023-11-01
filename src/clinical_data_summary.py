import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import glob
from utils import load_config


def therapy_modality_counts(data_master_df: pd.DataFrame) -> pd.Series:
    df = data_master_df.drop_duplicates(subset=['Cancer Type', 'Experimental', 'Control',
                                                'Trial ID', 'Trial Name'])
    return (df['Experimental Class'] + " + " + df['Control Class']).value_counts(ascending=True)


def plot_therapy_modality_counts(data: pd.DataFrame) -> plt.figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax = data.plot.barh(ax=ax, color=sns.color_palette('deep')[0])
    ax.set_xlabel('Count')
    ax.set_xticks(range(0, 20, 5))
    return fig


def cancer_type_counts(data_master_df: pd.DataFrame) -> pd.Series:
    df = data_master_df.drop_duplicates(subset=['Cancer Type', 'Experimental', 'Control',
                                                'Trial ID', 'Trial Name'])
    return df['Cancer Type'].value_counts(ascending=True)


def plot_cancer_type_counts(data: pd.DataFrame) -> plt.figure:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = data.plot.barh(ax=ax, color=sns.color_palette('deep')[0])
    ax.set_xlabel('Count')
    ax.set_xticks(range(0, 20, 5))
    return fig


def correlation_distribution(data_master_df: pd.DataFrame) -> pd.DataFrame:
    df = data_master_df.drop_duplicates(subset=['Cancer Type', 'Experimental', 'Control',
                                                'Trial ID', 'Trial Name'])
    return df[['Cancer Type', 'Experimental', 'Control',
               'Measured Combo for Correlation', 'Correlation Source Data', 'Pearsonr']]


def plot_correlation_distribution(data: pd.DataFrame) -> plt.figure:
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

    # plot correlation distribution
    correlation_distribution_df = correlation_distribution(data_master_df)
    correlation_distribution_df.to_csv(
        f'{config["main_combo"]["table_dir"]}/correlation_distribution_histplot.source_data.csv')
    fig3 = plot_correlation_distribution(correlation_distribution_df)
    fig3.savefig(f'{config["main_combo"]["fig_dir"]}/correlation_distribution_histplot.pdf',
                 bbox_inches='tight')


if __name__ == '__main__':
    main()
