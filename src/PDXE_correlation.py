import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from collections.abc import Iterable
from utils import load_config


def get_pdx_corr_data(df: pd.DataFrame, model_info: pd.DataFrame,
                      drug1: str, drug2: str, metric='BestAvgResponse'):
    """Prepare data to calculate correlation between drug response to drug1 and 2.

    Args:
        df (pd.DataFrame): drug response data
        model_info (pd.DataFrame): tumor type dataframe (model, tumor type info)
        drug1 (str): name of drug 1
        drug2 (str): name of drug 2
        metric (str, optional): drug response metric. Defaults to 'BestAvgResponse'.

    Returns:
        pd.DataFrame: 
    """
    a = df[df['Treatment'] == drug1].set_index('Model')[metric].astype(float)
    b = df[df['Treatment'] == drug2].set_index('Model')[metric].astype(float)
    assert a.index.is_unique
    assert b.index.is_unique
    assert model_info.index.is_unique
    merged = pd.concat([a, b, model_info], axis=1, join='inner')
    merged.columns = [drug1, drug2] + list(model_info.columns)
    return merged


def draw_corr_pdx(processed_df):
    """Plot scatterplot of two drug responses and calculate spearmanr correlation.

    Args:
        processed_df (pd.DataFrame): drug response data

    Returns:
        plt.figure: plotted figure
    """
    plt.style.use('env/publication.mplstyle')
    drug1 = processed_df.columns[0]
    drug2 = processed_df.columns[1]
    #r, p = pearsonr(processed_df[drug1], processed_df[drug2])
    r, p = spearmanr(processed_df[drug1], processed_df[drug2])
    fig, ax = plt.subplots(figsize=(3, 3))

    drug1_inactive = (processed_df[drug1] >= 30).sum()
    drug2_inactive = (processed_df[drug2] >= 30).sum()
    total = processed_df.shape[0]
    sns.scatterplot(x=drug1, y=drug2, data=processed_df, ax=ax)
    ax.axvline(0, color='gray', linestyle='--')
    ax.axhline(0, color='gray', linestyle='--')
    if (drug1_inactive/total > 0.8) or (drug2_inactive/total > 0.8):
        print("WARNING: at least one of the drug is inactive!")
    ax.set_xlabel(f'{drug1} BestAvgResponse')
    ax.set_ylabel(f'{drug2} BestAvgResponse')
    ax.set_title('n={0} r={1:.2f}'.format(total, r))

    return fig


def main():
    config = load_config()
    data_dir = config['PDX']['data_dir']
    fig_dir = config['PDX']['fig_dir']
    
    response = pd.read_csv(f'{data_dir}/PDXE_drug_response.csv', header=0)
    model_info = pd.read_csv(f'{data_dir}/PDXE_model_info.csv', header=0, index_col=0)
    df = get_pdx_corr_data(response, model_info, 'binimetinib', 'encorafenib')
    fig = draw_corr_pdx(df[(df['Tumor Type'] == 'CM') & (df['BRAF_mut'] == 1)])
    fig.savefig(f'{fig_dir}/CM-BRAFmut_binimetinib_encorafenib.pdf',
                bbox_inches='tight', pad_inches=0.1)

    df = get_pdx_corr_data(response, model_info, 'cetuximab', '5FU')
    fig = draw_corr_pdx(df[(df['Tumor Type'] == 'CRC') & (df['RAS_mut'] == 0)])
    fig.savefig(f'{fig_dir}/CRC-RASwt_cetuximab_5FU.pdf',
                bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()
