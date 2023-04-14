import pandas as pd
import shutil
import yaml

with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

raw_dir = CONFIG['cell_line']['raw_dir']
data_dir = CONFIG['cell_line']['data_dir']

def create_drug_info_df() -> pd.DataFrame:
    drug_info = pd.read_excel(f'{raw_dir}/Table S3_Screened Drug Info_Ling et al_2018.xlsx', 
                            engine='openpyxl', dtype=str)

    CTRPv2_drug_info = drug_info[drug_info['Database'] == 'CTRPv2']
    return CTRPv2_drug_info
    

def create_cell_info_df() -> pd.DataFrame:
    cell_info = pd.read_excel(f'{raw_dir}/Table S2_Screened Cell Line Info_Ling et al_2018.xlsx',
                            engine='openpyxl', dtype=str)

    cell_info = cell_info[(cell_info['Dataset'] == 'CTRPv2') & (cell_info['Drug Data Available in Dataset'] == 'y')]

    # make abbreviated names for cancer types
    cell_info['Cancer_Type_HH'] = cell_info['Condensed_Cancer_Type'].str.replace(' cancer', '').str.capitalize()
    for disease in ['Leukemia', 'Lymphoma', 'Myeloma']:
        cell_info.loc[cell_info['Cancer Statistics 2017 Primary Category'] == disease, 'Cancer_Type_HH'] = disease
    cell_info.loc[cell_info['Condensed_Cancer_Type'] == 'stomach cancer', 'Cancer_Type_HH'] = 'Gastric'
    cell_info.loc[cell_info['Disease Name'] == 'Head and neck squamous cell carcinoma', 'Cancer_Type_HH'] = 'HeadNeck'
    cell_info.loc[cell_info['Cancer Statistics 2017 Sub-Category'] == 'Melanoma of the skin', 'Cancer_Type_HH'] = 'Melanoma'
    cell_info.loc[cell_info['Cancer Statistics 2017 Sub-Category'].isin(['Colon', 'Rectum']), 'Cancer_Type_HH'] = 'Colorectal'
    cell_info.loc[cell_info['Disease Name'].str.contains('enal cell carcinoma'), 'Cancer_Type_HH'] = 'Renal'
    return cell_info


def main():
    drug_info = create_drug_info_df()
    cell_info = create_cell_info_df()
    drug_info.to_csv(f'{data_dir}/CTRPv2_drug.csv', index=False)
    cell_info.to_csv(f'{data_dir}/CTRPv2_CCL.csv', index=False)
    shutil.copy(f'{raw_dir}/Recalculated_CTRP_12_21_2018.txt', f'{data_dir}/Recalculated_CTRP_12_21_2018.txt')


if __name__ == '__main__':
    main()
