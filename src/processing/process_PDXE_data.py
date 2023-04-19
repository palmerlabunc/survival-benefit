import pandas as pd
import numpy as np
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = CONFIG['PDX']['raw_dir']
DATA_DIR = CONFIG['PDX']['data_dir']


def process_pdx_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Import Gao et al. (2015) PDX suppl. data.

    Returns:
        (pd.DataFrame, pd.DataFrame): a tuple of drug response dataframe and cancer type info dataframe.
    """
    info = pd.read_excel(f'{RAW_DIR}/Gao2015_suppl_table.xlsx',
                         sheet_name='PCT raw data',
                         engine='openpyxl', dtype=str)

    dat = pd.read_excel(f'{RAW_DIR}/Gao2015_suppl_table.xlsx',
                        sheet_name='PCT curve metrics',
                        engine='openpyxl', dtype=str)

    mut = pd.read_excel(f'{RAW_DIR}/Gao2015_suppl_table.xlsx',
                        sheet_name='pdxe_mut_and_cn2',
                        engine='openpyxl', dtype=str)

    info = info[['Model', 'Tumor Type', 'Treatment']].drop_duplicates()
    info = info.sort_values('Tumor Type')
    tumor_types = info[['Model', 'Tumor Type']].drop_duplicates(subset='Model').set_index('Model')
    braf_mut = np.intersect1d(
        tumor_types.index, mut[mut['Gene'] == 'BRAF']['Sample'].unique())
    ras_mut = np.intersect1d(
        tumor_types.index, mut[mut['Gene'].isin(['KRAS', 'NRAS'])]['Sample'].unique())

    tumor_types.loc[:, 'BRAF_mut'] = 0
    tumor_types.loc[:, 'RAS_mut'] = 0
    tumor_types.loc[braf_mut, 'BRAF_mut'] = 1
    tumor_types.loc[ras_mut, 'RAS_mut'] = 1
    return (dat, tumor_types)


def main():
    dat, tumor_types = process_pdx_data()
    dat.to_csv(f'{DATA_DIR}/PDXE_drug_response.csv', index=False)
    tumor_types.to_csv(f'{DATA_DIR}/PDXE_model_info.csv')


if __name__ == '__main__':
    main()