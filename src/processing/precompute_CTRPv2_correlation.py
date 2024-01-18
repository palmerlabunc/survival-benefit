import pandas as pd
import numpy as np
from .processing_utils import load_config


def filter_cancer_cells(ctrp: pd.DataFrame, cell_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    non_cancer = cell_info[cell_info['Cancer_Type_HH'] == 'Non-cancer'].index
    cell_info = cell_info.drop(non_cancer)
    cancer_type_info = cell_info['Cancer_Type_HH'].squeeze()
    ctrp = ctrp[~ctrp['Harmonized_Cell_Line_ID'].isin(non_cancer)]

    return ctrp, cell_info, cancer_type_info


def filter_active_and_clinical_drugs(ctrp: pd.DataFrame, drug_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Keep only active drugs
    grouped = ctrp.groupby('Harmonized_Compound_Name')
    groups = np.array([name for name, unused_df in grouped])
    active_drugs = groups[grouped['CTRP_AUC'].quantile(0.1) < 0.8]
    ctrp = ctrp[ctrp['Harmonized_Compound_Name'].isin(active_drugs)]
    drug_info = drug_info[drug_info['Harmonized Name'].isin(active_drugs)]

    # Keep only drugs in clinical phases
    clinical_drugs = drug_info[~drug_info['Clinical Phase'].isin(
        ['Withdrawn', 'Preclinical', np.nan])]['Harmonized Name']
    ctrp = ctrp[ctrp['Harmonized_Compound_Name'].isin(clinical_drugs)]
    drug_info = drug_info[drug_info['Harmonized Name'].isin(clinical_drugs)]

    return ctrp, drug_info


def import_ctrp_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Imports all CTRP data. Filter for active & clinical stage drugs and cancer cell lines.

    Args:
        data_dir (str): data directory

    Returns:
        pd.DataFrame: cell line information
        pd.DataFrame: drug information
        pd.Series: cancer type information
        pd.DataFrame: viablility data
    """

    # metadata
    cell_info = pd.read_csv(
        f'{data_dir}/CTRPv2_CCL.csv', index_col=0)
    drug_info = pd.read_csv(
        f'{data_dir}/CTRPv2_drug.csv', index_col=None)
    # viability data
    ctrp = pd.read_csv(f'{data_dir}/Recalculated_CTRP_12_21_2018.txt',
                       sep='\t', index_col=0)

    # scale AUC
    ctrp.loc[:, 'CTRP_AUC'] = ctrp['CTRP_AUC'] / 16

    # drop non-cancer cell lines
    ctrp, cell_info, cancer_type_info = filter_cancer_cells(ctrp, cell_info)

    # keep only active & clinical phase drugs
    ctrp, drug_info = filter_active_and_clinical_drugs(ctrp, drug_info)

    return cell_info, drug_info, cancer_type_info, ctrp


def precompute_correlation(ctrp: pd.DataFrame, 
                           cancer_type_info: pd.Series, 
                           cancer_type=None,
                           corr_method='pearson') -> pd.DataFrame:
    """Return correlation matrix dataframe for all drug pairs available for either pan-cancer
    or a certain cancer type. At least min_ccl_cnt number of cell lines are required.

    Args:
        ctrp (pd.DataFrame): ctrp data from import_ctrp_data
        cancer_type_info (pd.Series): ctrp data from import_ctrp_data
        cancer_type (str, optional): Cancer_Type_HH for cancer type-specfic corrrelation. If None, use pan-cancer. Defaults to None.
        corr_method (str, optional): correlation method {'pearson', 'kendall', 'spearman'}. Defaults to 'pearson'.

    Returns:
        pd.DataFrame or None: correlation matrix dataframe or None if it cannot be calculated
    """

    min_ccl_cnt = 20
    tmp = ctrp.drop_duplicates(
        ['Harmonized_Cell_Line_ID', 'Harmonized_Compound_Name'])
    ctrp_wide = tmp.pivot(index='Harmonized_Cell_Line_ID',
                          columns='Harmonized_Compound_Name', values='CTRP_AUC')

    available_cancer_types = cancer_type_info.unique()
    cancer_type_counts = cancer_type_info.value_counts()
   
    if cancer_type is not None:
        can_do_cancer_specific = (cancer_type in available_cancer_types) and (
            cancer_type_counts[cancer_type] > min_ccl_cnt)
        if can_do_cancer_specific:
            ccls = cancer_type_info[cancer_type_info == cancer_type].index
            cancer_specific_ctrp = ctrp_wide.reindex(ccls)
            # filter for drugs that have at least 20% of celll ines with AUC < 0.8
            cancer_specific_ctrp = cancer_specific_ctrp.loc[:, cancer_specific_ctrp.quantile(0.2) < 0.8]
            corr_df = cancer_specific_ctrp.corr(corr_method, min_periods=min_ccl_cnt)
        else:
            # don't need to filter for active drugs here because we have already done so when importing ctrp
            print(f"Cannot calculate {cancer_type}-specific correlation")
            return None

    else: # pan-cancer
        corr_df = ctrp_wide.corr(corr_method, min_periods=min_ccl_cnt)

    # drop drugs where correlation cannot be calculated
    corr_df = corr_df.dropna(axis=0, how='all').dropna(axis=1, how='all').round(5)
    return corr_df


def precompute_moa_drug_list(drug_info: pd.DataFrame) -> tuple[np.array, np.array]:
    """_summary_

    Args:
        drug_info (pd.DataFrame): _description_

    Returns:
        np.array: Chemotherapy drug list
        np.array: Targeted therapy drug list
    """
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
    chemo_drugs = drug_info[drug_info['MOA'].isin(cyto_moa)]['Harmonized Name'].values
    targeted_drugs = drug_info[drug_info['MOA'].isin(
        targ_moa)]['Harmonized Name'].values
    return chemo_drugs, targeted_drugs


def main():
    config = load_config()
    data_dir = config['cell_line']['data_dir']
    cell_info, drug_info, cancer_type_info, ctrp = import_ctrp_data(data_dir)

    corr_method = config['cell_line']['corr_method']
    corr_df = precompute_correlation(ctrp, cancer_type_info, corr_method=corr_method)
    corr_df.to_csv(f'{data_dir}/PanCancer_all_pairwise_{corr_method}_correlation.csv')

    cancer_types = cancer_type_info.unique()
    for cancer in cancer_types:
        corr_df = precompute_correlation(ctrp, cancer_type_info, cancer, corr_method=corr_method)
        if corr_df is not None:
            corr_df.index.name = None
            corr_df.columns.name = None
            corr_df.to_csv(f'{data_dir}/{cancer}_all_pairwise_{corr_method}_correlation.csv')


if __name__ == '__main__':
    main()
