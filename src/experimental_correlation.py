import pandas as pd
import numpy as np
from itertools import combinations
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

plt.style.use('env/publication.mplstyle')

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

EXPERIMENTAL_DATA_DIR = CONFIG['cell_line']['data_dir']
FIG_DIR = CONFIG['fig_dir']
TABLE_DIR = CONFIG['table_dir']


def get_ctrp_corr_data(df: pd.DataFrame, cancer_type: pd.Series, 
                       drug_a: str, drug_b: str, metric='CTRP_AUC'):
    """Filters and returns data frame ready to calculate correlation

    Args:
        df (pd.DataFrame): CTRPv2 viability data
        cancer_type (pd.Series): cancer type information
        drug_a (str): name of drug A
        drug_b (str): naem of drug B
        metric (str): viability metric to use (Default: CTRP_AUC)

    Returns:
        pd.DataFrame: ready to calculate correlation.
    """
    a = df[df['Harmonized_Compound_Name'] ==
           drug_a][['Harmonized_Cell_Line_ID', metric]]
    b = df[df['Harmonized_Compound_Name'] ==
           drug_b][['Harmonized_Cell_Line_ID', metric]]
    # take mean if duplicated cell lines
    a = a.groupby('Harmonized_Cell_Line_ID').mean()
    b = b.groupby('Harmonized_Cell_Line_ID').mean()
    merged = pd.concat([a, b, cancer_type], axis=1, join='inner')
    merged.columns = [drug_a, drug_b, 'Cancer_Type_HH']

    # drop cancer type if 1st quantile is less then 0.8 both drugs
    by_type = merged.groupby('Cancer_Type_HH').quantile(0.25)
    valid_types = by_type[(by_type < 0.8).sum(axis=1) != 0].index
    merged = merged[merged['Cancer_Type_HH'].isin(valid_types)]
    return merged


def get_pdx_corr_data(df: pd.DataFrame, tumor_types: pd.DataFrame, 
                      drug1: str, drug2: str, metric='BestAvgResponse'):
    """Prepare data to calculate correlation between drug response to drug1 and 2.

    Args:
        df (pd.DataFrame): drug response data
        tumor_types (pd.DataFrame): tumor type dataframe (model, tumor type info)
        drug1 (str): name of drug 1
        drug2 (str): name of drug 2
        metric (str, optional): drug response metric. Defaults to 'BestAvgResponse'.

    Returns:
        pd.DataFrame: 
    """    
    a = df[df['Treatment'] == drug1].set_index('Model')[metric].astype(float)
    b = df[df['Treatment'] == drug2].set_index('Model')[metric].astype(float)
    merged = pd.concat([a, b, tumor_types], axis=1, join='inner')
    merged.columns = [drug1, drug2, 'Tumor Type']
    return merged


def draw_corr_pdx(df: pd.DataFrame, tumor_types: pd.DataFrame, 
                  drug1: str, drug2: str, metric='BestAvgResponse'):
    """Plot scatterplot of two drug responses and calculate spearmanr correlation.

    Args:
        df (pd.DataFrame): drug response data
        tumor_types (pd.DataFrame): tumor type data
        drug1 (str): name of drug 1
        drug2 (str): name of drug 2
        metric (str, optional): drug response metrics. Defaults to 'BestAvgResponse'.

    Returns:
        plt.figure: plotted figure
    """    
    tmp = get_pdx_corr_data(df, tumor_types, drug1, drug2, metric=metric)
    r, p = spearmanr(tmp[drug1], tmp[drug2])
    fig, ax = plt.subplots(figsize=(2, 2))

    a = df[df['Treatment'] == drug1]['ResponseCategory'] == 'PD'
    b = df[df['Treatment'] == drug2]['ResponseCategory'] == 'PD'
    if metric == 'BestAvgResponse':
        ax.axvline(0, color='gray', linestyle='--')
        ax.axhline(0, color='gray', linestyle='--')
        if (a.sum() / a.shape[0] > 0.75) or (b.sum() / b.shape[0] > 0.75):
            print("WARNING: at least one of the drug is inactive!")
    tumor = tmp['Tumor Type'].unique()
    if tumor.size > 1:
        sns.scatterplot(x=drug1, y=drug2, hue='Tumor Type', data=tmp, ax=ax)
        ax.set_title('n={0} rho={1:.2f}'.format(tmp.shape[0], r))
    else:
        sns.scatterplot(x=drug1, y=drug2, data=tmp, ax=ax)
        ax.set_title('{0} n={1} rho={2:.2f}'.format(tumor[0], tmp.shape[0], r))
    ax.set_xlabel(drug1 + ' ({})'.format(metric))
    ax.set_ylabel(drug2 + ' ({})'.format(metric))

    return fig


def draw_ctrp_spearmanr_distribution(all_pairs: np.array, cyto_pairs: np.array, targ_pairs: np.array, cyto_targ_pairs: np.array):
    """Plot histograms (distributions) of spearmanr correlations between CTRPv2 drug pairs.

    Args:
        all_pairs (np.ndarray): 1-D array of pairwise correlation values between all drug pairs
        cyto_pairs (np.ndarray): 1-D array of pairwise correlation values between cytotoxic drug pairs
        targ_pairs (np.ndarray): 1-D array of pairwise correlation values between targeted drug pairs
        cyto_targ_pairs (np.ndarray): 1-D array of pairwise correlation values between cytotoxic and targted drugs

    Returns:
        plt.figure: plotted figure
    """    
    fig, ax = plt.subplots(figsize=(3, 2), dpi=300)

    sns.despine()
    sns.histplot(all_pairs, ax=ax, label='all pairs', color=sns.color_palette()[0])
    sns.histplot(targ_pairs, label='targeted pairs', color=sns.color_palette()[2])
    sns.histplot(cyto_targ_pairs, label='cytotoxic-targeted pairs',
                color=sns.color_palette()[3])
    sns.histplot(cyto_pairs, label='cytotoxic pairs', color=sns.color_palette()[1])
    ax.set_xlim(-0.5, 1)
    ax.set_xlabel('Spearman rho')
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),)

    return fig


def draw_corr_cell(ctrp_df, cancer_type, drug1, drug2, 
                   metric='CTRP_AUC', only_cancer_type=None):
    """Plot scatterplot of two drug responses and calculate spearmanr correlation.

    Args:
        ctrp_df (pd.DataFrame): CTRPv2 viability data
        cancer_type (pd.Series): cancer type information
        drug1 (str): name of drug 1
        drug2 (str): name of drug 2
        metric (str, optional): drug response metric. Defaults to 'CTRP_AUC'.
        only_cancer_type (str, optional): Specific cancer type to use. 
            Default uses all cancer types where the drug is active. Defaults to None.

    Returns:
        plt.figure: plotted figure
    """    
    dat = get_ctrp_corr_data(ctrp_df, cancer_type, drug1, drug2, metric)
    if only_cancer_type is not None:
        dat = dat[dat['Cancer_Type_HH'] == only_cancer_type]
    r, p = spearmanr(dat[drug1], dat[drug2], axis=0, nan_policy='omit')

    ticks = [0, 0.5, 1]

    fig, ax = plt.subplots(figsize=(2, 2), constrained_layout=True, 
                           subplot_kw=dict(box_aspect=1))
    sns.despine()
    sns.scatterplot(x=drug1, y=drug2, size=1, color='k', alpha=0.7,
                    data=dat, ax=ax)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel(drug1 + ' ' + metric)
    ax.set_ylabel(drug2 + ' ' + metric)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # identity line
    lims = [0, min(ax.get_xlim()[1], ax.get_ylim()[1])]
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    ax.set_title('n={}, rho={:.2f}'.format(dat.shape[0], r))
    
    return fig


def import_ctrp_data():
    """Imports all CTRP data

    Returns:
        pd.DataFrame: cell line information
        pd.DataFrame: drug information
        pd.Series: cancer type information
        pd.DataFrame: viablility data
        pd.DataFrame: pairwise correlation data
    """
    # metadata
    cell_info = pd.read_csv(
        f'{EXPERIMENTAL_DATA_DIR}/CTRPv2_CCL.csv', index_col=0)
    drug_info = pd.read_csv(
        f'{EXPERIMENTAL_DATA_DIR}/CTRPv2_drug.csv', index_col=None)
    # drop non-cancer cell types
    noncancer = cell_info[cell_info['Cancer_Type_HH'] == 'Non-cancer'].index
    cell_info = cell_info.drop(noncancer)
    cancer_type = cell_info['Cancer_Type_HH'].squeeze()

    # viability data
    ctrp = pd.read_csv(f'{EXPERIMENTAL_DATA_DIR}/Recalculated_CTRP_12_21_2018.txt',
                       sep='\t', index_col=0)

    # scale AUC
    ctrp.loc[:, 'CTRP_AUC'] = ctrp['CTRP_AUC'] / 16
    # drop non-cancer cell lines
    ctrp = ctrp[~ctrp['Harmonized_Cell_Line_ID'].isin(noncancer)]

    # keep only active drugs
    grouped = ctrp.groupby('Harmonized_Compound_Name')
    groups = np.array([name for name, unused_df in grouped])
    active_drugs = groups[grouped['CTRP_AUC'].quantile(0.1) < 0.8]
    ctrp = ctrp[ctrp['Harmonized_Compound_Name'].isin(active_drugs)]
    drug_info = drug_info[drug_info['Harmonized Name'].isin(active_drugs)]

    # keep only drugs in clinical phases
    clinical_drugs = drug_info[~drug_info['Clinical Phase'].isin(
        ['Withdrawn', 'Preclinical', np.nan])]['Harmonized Name']
    ctrp = ctrp[ctrp['Harmonized_Compound_Name'].isin(clinical_drugs)]
    drug_info = drug_info[drug_info['Harmonized Name'].isin(clinical_drugs)]
    return cell_info, drug_info, cancer_type, ctrp


def prepare_ctrp_agg_data(drug_info: pd.DataFrame):
    """Prepare data frame to plot distribution of correlations for drug pairs.

    Args:
        drug_info (pd.DataFrame): drug info data

    Returns:
        np.array: array of correlation values between all drug pairs
        np.array: array of correlation values between cytotoxic drug pairs
        np.array: array of correlation values between targeted drug pairs
        np.array: array of correlation values between cytotoxic-targeted drug pairs
    """
    df = pd.read_csv(f'{EXPERIMENTAL_DATA_DIR}/CTRPv2_clincal_active_drug_pairwise_corr.csv',
                     index_col=0)
    vals = df.values.flatten()
    vals = vals[~np.isnan(vals)]

    # referenced from http://www.bccancer.bc.ca/pharmacy-site/Documents/Pharmacology_Table.pdf
    cyto_moa = ['DNA alkylating agent|DNA inhibitor',
                'DNA alkylating agent|DNA synthesis inhibitor',
                'DNA synthesis inhibitor',
                'ribonucleotide reductase inhibitor',
                'thymidylate synthase inhibitor',
                'dihydrofolate reductase inhibitor',
                'DNA alkylating agent',
                'DNA inhibitor',
                'topoisomerase inhibitor',
                'tubulin polymerization inhibitor',
                'src inhibitor|tubulin polymerization inhibitor',
                'HDAC inhibitor',
                'DNA methyltransferase inhibitor']

    targ_moa = ['src inhibitor',
                'MEK inhibitor',
                'EGFR inhibitor',
                'Abl kinase inhibitor|Bcr-Abl kinase inhibitor',
                'ALK tyrosine kinase receptor inhibitor',
                'RAF inhibitor|VEGFR inhibitor',
                'FLT3 inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|RAF inhibitor|RET tyrosine kinase inhibitor|VEGFR inhibitor',
                'EGFR inhibitor|RET tyrosine kinase inhibitor|VEGFR inhibitor',
                'NFkB pathway inhibitor|proteasome inhibitor',
                'mTOR inhibitor',
                'mTOR inhibitor|PI3K inhibitor',
                'Bcr-Abl kinase inhibitor|ephrin inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|src inhibitor|tyrosine kinase inhibitor',
                'FLT3 inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|RET tyrosine kinase inhibitor|VEGFR inhibitor',
                'BCL inhibitor',
                'KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|VEGFR inhibitor',
                'PDGFR tyrosine kinase receptor inhibitor|VEGFR inhibitor',
                'PLK inhibitor',
                'Abl kinase inhibitor|Bcr-Abl kinase inhibitor|src inhibitor',
                'PI3K inhibitor',
                'AKT inhibitor',
                'BCL inhibitor|MCL1 inhibitor',
                'FLT3 inhibitor|JAK inhibitor',
                'RAF inhibitor',
                'Aurora kinase inhibitor|Bcr-Abl kinase inhibitor|FLT3 inhibitor|JAK inhibitor',
                'NFkB pathway inhibitor',
                'RET tyrosine kinase inhibitor|VEGFR inhibitor',
                'VEGFR inhibitor',
                "Bruton's tyrosine kinase (BTK) inhibitor",
                'CDK inhibitor',
                'CHK inhibitor',
                'FGFR inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|RAF inhibitor|RET tyrosine kinase inhibitor|VEGFR inhibitor',
                'KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|src inhibitor',
                'KIT inhibitor|VEGFR inhibitor',
                'proteasome inhibitor',
                'Abl kinase inhibitor|Aurora kinase inhibitor|FLT3 inhibitor',
                'CDK inhibitor|cell cycle inhibitor|MCL1 inhibitor',
                'cell cycle inhibitor|PLK inhibitor',
                'FGFR inhibitor',
                'FGFR inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor|VEGFR inhibitor',
                'FGFR inhibitor|PDGFR tyrosine kinase receptor inhibitor|VEGFR inhibitor',
                'FLT3 inhibitor|KIT inhibitor|PDGFR tyrosine kinase receptor inhibitor',
                'JAK inhibitor']

    cyto_drugs = drug_info[drug_info['MOA'].isin(
        cyto_moa)]['Harmonized Name'].values
    targ_drugs = drug_info[drug_info['MOA'].isin(
        targ_moa)]['Harmonized Name'].values

    # cytotoxic - cytotoxic pairs
    cyto_vals = []
    for i, j in combinations(cyto_drugs, 2):
        cyto_vals.append(df.at[i, j])
    cyto_vals = np.array(cyto_vals)

    # targeted - targeted pairs
    targ_vals = []
    for i, j in combinations(targ_drugs, 2):
        targ_vals.append(df.at[i, j])
    targ_vals = np.array(targ_vals)
    targ_vals = targ_vals[~np.isnan(targ_vals)]

    # targeted - cytotoxic pairs
    mesh = np.array(np.meshgrid(targ_drugs, cyto_drugs))
    combi = mesh.T.reshape(-1, 2)
    cyto_targ_pairs = []
    for i, j in combi:
        v = df.at[i, j]
        if np.isnan(v):
            v = df.at[j, i]
        cyto_targ_pairs.append(v)
    cyto_targ_pairs = np.array(cyto_targ_pairs)
    cyto_targ_pairs = cyto_targ_pairs[~np.isnan(cyto_targ_pairs)]
    return vals, cyto_vals, targ_vals, cyto_targ_pairs


def import_pdx_data():
    """Import Gao et al. (2015) PDX suppl. data.

    Returns:
        (pd.DataFrame, pd.DataFrame): a tuple of drug response dataframe and cancer type info dataframe.
    """
    info = pd.read_excel(f'{EXPERIMENTAL_DATA_DIR}/Gao2015_suppl_table.xlsx',
                         sheet_name='PCT raw data',
                         engine='openpyxl', dtype=str)

    dat = pd.read_excel(f'{EXPERIMENTAL_DATA_DIR}/Gao2015_suppl_table.xlsx',
                        sheet_name='PCT curve metrics',
                        engine='openpyxl', dtype=str)

    info = info[['Model', 'Tumor Type', 'Treatment']].drop_duplicates()
    info = info.sort_values('Tumor Type')
    tumor_types = info[['Model', 'Tumor Type']
                       ].drop_duplicates().set_index('Model').iloc[:-1, :]
    return (dat, tumor_types)


def get_distribution_report(all_pairs: np.array, cyto_pairs: np.array,
                            targ_pairs: np.array, cyto_targ_pairs: np.array):
    df = pd.DataFrame(np.nan, index=['all_pairs', 'cytotoxic_pairs', 'targeted_pairs', 'cyto+targeted_pairs'],
                      columns=['N', 'mean', 'sd', '95% lower range', '95% lower range'])
    pair_list = [all_pairs, cyto_pairs, targ_pairs, cyto_targ_pairs]
    for i in range(len(pair_list)):
        pair = pair_list[i]
        df.iat[i, 0] = len(pair)
        df.iat[i, 1] = np.mean(pair)
        df.iat[i, 2] = np.std(pair)
        df.iat[i, 3] = np.quantile(pair, 0.025)
        df.iat[i, 4] = np.quantile(pair, 0.975)
    return df.round(3)


def main():
    pdx_df, tumor_types = import_pdx_data()
    drug1, drug2 = 'cetuximab', '5FU'
    fig = draw_corr_pdx(pdx_df, tumor_types, drug1, drug2)
    fig.savefig(f'{FIG_DIR}/CRC_{drug1}_{drug2}_BestAvgResponse_corr.pdf',
                bbox_inches='tight', pad_inches=0.1)

    # use cell line data (CTRPv2)
    cell_info, drug_info, cancer_type, ctrp = import_ctrp_data()
    drug_pairs = [('5-Fluorouracil', 'Docetaxel'), ('5-Fluorouracil', 'Lapatinib'),
                  ('Gemcitabine', 'Oxaliplatin'), ('Methotrexate', 'Oxaliplatin'),
                  ('5-Fluorouracil', 'Topotecan'), ('5-Fluorouracil', 'Oxaliplatin'),
                  ('nintedanib', 'Docetaxel'), ('Ifosfamide', 'Doxorubicin'),
                  ('Lapatinib', 'Paclitaxel'), ('Selumetinib', 'Dacarbazine')]
    for drug1, drug2 in drug_pairs:
        fig = draw_corr_cell(ctrp, cancer_type, drug1, drug2)
        fig.savefig(f'{FIG_DIR}/{drug1}_{drug2}_AUC_corr.pdf')

    for drug1, drug2 in [('Dabrafenib', 'Trametinib')]:
        fig = draw_corr_cell(ctrp, cancer_type, drug1,
                             drug2, only_cancer_type='Melanoma')
        fig.savefig(f'{FIG_DIR}/{drug1}_{drug2}_AUC_corr.pdf')

    # cell line drug pair spearmanr distribution
    all_pairs, cyto_pairs, targ_pairs, cyto_targ_pairs = prepare_ctrp_agg_data(
        drug_info)
    fig = draw_ctrp_spearmanr_distribution(all_pairs, cyto_pairs,
                                           targ_pairs, cyto_targ_pairs)
    fig.savefig(f'{FIG_DIR}/CTRPv2_corr_distributions.pdf',
                bbox_inches='tight', pad_inches=0.1)

    dist_report = get_distribution_report(
        all_pairs, cyto_pairs, targ_pairs, cyto_targ_pairs)
    dist_report.to_csv(f'{TABLE_DIR}/experimental_correlation_report.csv')


if __name__ == '__main__':
    main()
