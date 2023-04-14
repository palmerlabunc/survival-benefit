import pandas as pd
import numpy as np
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = CONFIG['cell_line']['data_dir']


def import_ctrp_data():
    """Imports all CTRP data. Filter for active & clinical stage drugs and cancer cell lines.

    Returns:
        pd.DataFrame: cell line information
        pd.DataFrame: drug information
        pd.Series: cancer type information
        pd.DataFrame: viablility data
        pd.DataFrame: pairwise correlation data
    """
    # metadata
    cell_info = pd.read_csv(
        f'{DATA_DIR}/CTRPv2_CCL.csv', index_col=0)
    drug_info = pd.read_csv(
        f'{DATA_DIR}/CTRPv2_drug.csv', index_col=None)
    # drop non-cancer cell types
    noncancer = cell_info[cell_info['Cancer_Type_HH'] == 'Non-cancer'].index
    cell_info = cell_info.drop(noncancer)
    cancer_type_info = cell_info['Cancer_Type_HH'].squeeze()

    # viability data
    ctrp = pd.read_csv(f'{DATA_DIR}/Recalculated_CTRP_12_21_2018.txt',
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
    return cell_info, drug_info, cancer_type_info, ctrp


def precompute_correlation(ctrp: pd.DataFrame, cancer_type_info: pd.Series, cancer_type=None) -> pd.DataFrame:
    """Return correlation matrix dataframe for all drug pairs available for either pan-cancer
    or a certain cancer type. At least min_ccl_cnt number of cell lines are required.

    Args:
        ctrp (pd.DataFrame): ctrp data from import_ctrp_data
        cancer_type_info (pd.Series): ctrp data from import_ctrp_data
        cancer_type (str, optional): Cancer_Type_HH for cancer type-specfic corrrelation. If None, use pan-cancer. Defaults to None.

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
            # filter for drugs that have at least 25% of celll ines with AUC < 0.8
            cancer_specific_ctrp = cancer_specific_ctrp.loc[:, cancer_specific_ctrp.quantile(0.25) < 0.8]
            corr_df = cancer_specific_ctrp.corr('pearson', min_periods=min_ccl_cnt)
        else:
            # don't need to filter for active drugs here because we have already done so when importing ctrp
            print(f"Cannot calculate {cancer_type}-specific correlation")
            return None

    else: # pan-cancer
        corr_df = ctrp_wide.corr('pearson', min_periods=min_ccl_cnt)

    # drop drugs where correlation cannot be calculated
    corr_df = corr_df.dropna(axis=0, how='all').dropna(axis=1, how='all').round(5)
    return corr_df
    

def main():
    cell_info, drug_info, cancer_type_info, ctrp = import_ctrp_data()

    corr_df = precompute_correlation(ctrp, cancer_type_info)
    corr_df.to_csv(f'{DATA_DIR}/PanCancer_all_pairwise_correlation.csv')

    cancer_types = cancer_type_info.unique()
    for cancer in cancer_types:
        corr_df = precompute_correlation(ctrp, cancer_type_info, cancer)
        if corr_df is not None:
            corr_df.index.name = None
            corr_df.columns.name = None
            corr_df.to_csv(f'{DATA_DIR}/{cancer}_all_pairwise_correlation.csv')


if __name__ == '__main__':
    main()
