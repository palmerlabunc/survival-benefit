from utils import load_config
import pandas as pd
import numpy as np
import argparse

def get_control_prefix(cancer_type, control_drug, 
                       first_author, year, endpoint, additional_info=None) -> str:
    prefix = f'{cancer_type}_{control_drug}_{first_author}{year}_{endpoint}'
    if additional_info is None:
        return prefix
    return f'{prefix}_{additional_info}'

def get_combination_prefix(cancer_type, experimental_drug, control_drug, 
                           first_author, year, endpoint, additional_info=None) -> str:
    prefix = f'{cancer_type}_{experimental_drug}-{control_drug}_{first_author}{year}_{endpoint}'
    if additional_info is None:
        return prefix
    return f'{prefix}_{additional_info}'


def get_label(cancer_type, experimental_drug, control_drug, first_author, year, endpoint, additional_info=None) -> str:
    if additional_info is None:
        return f'{cancer_type} {experimental_drug} + {control_drug} ({first_author} et al. {year}) {endpoint}'
    return f'{cancer_type} ({additional_info}) {experimental_drug} + {control_drug} ({first_author} et al. {year}) {endpoint}'


def parse_trial_key(key: str) -> dict:
    """Parse the trial key to get the information about the trial.
    (e.g. Breast_Lapatinib-Letrozole_Schwartzberg2010)

    Args:
        key (str): The trial key

    Returns:
        dict: A dictionary containing the information about the trial
    """
    additional_info = None
    endpoint = None
    contains_endpoint= False
    # does the key contain endpint?
    tokens = key.split('_')
    for token in tokens:
        if token in ['OS', 'PFS', 'TTP', 'RFS', 'EFS']:
            contains_endpoint = True
            break
    
    if contains_endpoint:
        if len(key.split('_')) == 5:
            cancer_type, drugs, author_year, endpoint, additional_info = key.split('_')
        else:
            cancer_type, drugs, author_year, endpoint = key.split('_')
    else:
        # this is when there is additional information (e.g. RAS WT)
        if len(key.split('_')) == 4:
            cancer_type, drugs, author_year, additional_info = key.split('_')
        else:
            cancer_type, drugs, author_year = key.split('_')

    
    experimental_drug, control_drug = drugs.split('-')

    return {'cancer_type': cancer_type, 'experimental_drug': experimental_drug, 
            'control_drug': control_drug, 'author_year': author_year, 
            'endpoint': endpoint, 'additional_info': additional_info}


def create_endpoint_subset_sheet(master_df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Returrns a subset of the master dataframe containing only the endpoint of interest.

    Args:
        master_df (pd.DataFrame): data master sheet; assumes that the index is key
        endpoint (str): 'OS' or 'Surrogate'

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: subset of the master dataframe containing only the endpoint of interest
    """    
    hr_related_cols = ['HR', 'HR 95% CI', 'median benefit', 'HR pval']
    if endpoint == 'OS':
        condition = (master_df['Include OS (0/1/NA)'] == 1)
        df = master_df[condition]
        df = df.drop([f'Surrogate {item}' for item in hr_related_cols], axis=1)

    elif endpoint == 'Surrogate':
        df = master_df[master_df['Include Surrogate (0/1/NA)'] == 1]
        df = df.drop([f'OS {item}' for item in hr_related_cols], axis=1)
    else:
        raise ValueError(f'Endpoint {endpoint} is not supported.')
    
    df = df.rename({f'{endpoint} HR': 'HR', f'{endpoint} HR 95% CI': 'HR 95% CI',
                    f'{endpoint} median benefit': 'median benefit', f'{endpoint} HR pval': 'HR pval'},
                    axis=1)

    df.loc[:, 'HR'] = df['HR'].astype(float)
    
    df.loc[:, 'HR_95low'] = df['HR 95% CI'].apply(lambda x: eval(x.split(',')[0]) if x is not np.nan else x)
    df.loc[:, 'HR_95high'] = df['HR 95% CI'].apply(lambda x: eval(x.split(',')[1]) if x is not np.nan else x)

    for key, row in df.iterrows():
        if endpoint == 'Surrogate':
            metric = row['Surrogate Metric']
        else:
            metric = endpoint
        trial_info_dict = parse_trial_key(key)

        if not isinstance(df.loc[key, 'median benefit'], float):
            try:
                df.loc[key, 'median benefit'] = float(row['median benefit'])
            except ValueError:
                df.loc[key, 'median benefit'] = np.nan
        
        new_key = f"{trial_info_dict['cancer_type']}_{trial_info_dict['experimental_drug']}-{trial_info_dict['control_drug']}_{trial_info_dict['author_year']}_{metric}"
        if trial_info_dict['additional_info'] is not None:
            new_key = f"{new_key}_{trial_info_dict['additional_info']}"
        df.loc[key, 'key_endpoint'] = new_key
    return df


def create_main_combo_input_sheet(master_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    raw_path = config['main_combo']['raw_dir']
    processed_path = config['main_combo']['data_dir']

    data = []
    for idx, row in master_df.iterrows():
        additional_info = None
        if len(idx.split('_')) == 4:  # this is when there is additional information (e.g. RAS WT)
            additional_info = idx.rsplit('_', 1)[-1]
        cancer_type = row['Cancer Type']
        experimental_drug = row['Experimental']
        control_drug = row['Control']
        first_author = row['First Author']
        try:
            year = int(row['Year'])
        except ValueError:
            print(idx)
        n_combo = int(row['Combination Arm N'])
        n_control = int(row['Control Arm N'])
        corr = row['Spearmanr']

        include_os = (row['Include OS (0/1/NA)'] == 1)
        include_surrogate = (row['Include Surrogate (0/1/NA)'] == 1)
        
        if include_os:
            data.append([raw_path, processed_path, 
                         get_control_prefix(cancer_type, control_drug, first_author, year, 'OS', additional_info=additional_info), 
                         get_combination_prefix(
                             cancer_type, experimental_drug, control_drug, first_author, year, 'OS', additional_info=additional_info),
                         get_label(cancer_type, experimental_drug, control_drug,
                                   first_author, year, 'OS', additional_info=additional_info),
                         n_control, n_combo, corr])
        
        if include_surrogate:
            endpoint = row['Surrogate Metric'].strip()
            data.append([raw_path, processed_path,
                         get_control_prefix(cancer_type, control_drug, first_author, year, endpoint, additional_info=additional_info),
                         get_combination_prefix(
                             cancer_type, experimental_drug, control_drug, first_author, year, endpoint, additional_info=additional_info),
                         get_label(cancer_type, experimental_drug, control_drug, first_author, year, endpoint, additional_info=additional_info),
                         n_control, n_combo, corr])
    
    columns = ['Raw Path', 'Processed Path', 'Control',
               'Combination', 'Label', 'N_Control', 'N_Combination', 'Corr']
    result = pd.DataFrame(data=data, columns=columns)
    return result


def create_biomarker_input_sheet(biomarker_data_sheet: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Args:
        biomarker_data_sheet (pd.DataFrame): _description_
        config (dict): _description_

    Returns:
        pd.DataFrame: _description_
    """
    data = []
    for idx, row in biomarker_data_sheet.iterrows():
        
        if row['Include in Main Figure'] == 0:
            continue

        additional_info = None
        if len(idx.split('_')) == 4:  # this is when there is additional information (e.g. RAS WT)
            additional_info = idx.rsplit('_', 1)[-1]
        
        cancer_type = row['Cancer Type']
        experimental_drug = row['Experimental']
        control_drug = row['Control']
        first_author = row['First Author']
        
        try:
            year = int(row['Year'])
        except ValueError:
            print(idx)
        
        n_combo = int(row['Combination Arm N'])
        n_control = int(row['Control Arm N'])
        corr = row['Spearmanr']
        biomarker = row['Biomarker']

        
        include_surrogate = (row['Include Surrogate (0/1/NA)'] == 1)
        has_surrogate_biomarker = pd.notna(row['Biomarker-positive Surrogate HR'])
        
        
        if include_surrogate and has_surrogate_biomarker:
            metric = 'Surrogate'
            ITT_HR = row[f'{metric} HR']
            ITT_HR_95low = row[f'{metric} HR 95% CI'].split(',')[0]
            ITT_HR_95high = row[f'{metric} HR 95% CI'].split(',')[1]
            biomarker_HR = row[f'Biomarker-positive {metric} HR']
            biomarker_HR_95low = row[f'Biomarker-positive {metric} HR 95% CI'].split(',')[0]
            biomarker_HR_95high = row[f'Biomarker-positive {metric} HR 95% CI'].split(',')[1]
        
            endpoint = row['Surrogate Metric'].strip()
            data.append([get_control_prefix(cancer_type, control_drug, first_author, year, endpoint, additional_info=additional_info),
                         get_combination_prefix(
                             cancer_type, experimental_drug, control_drug, first_author, year, endpoint, additional_info=additional_info),
                         get_label(cancer_type, experimental_drug, control_drug, first_author, year, endpoint, additional_info=additional_info),
                         n_control, n_combo, corr, ITT_HR, ITT_HR_95low, ITT_HR_95high, 
                         biomarker, biomarker_HR, biomarker_HR_95low, biomarker_HR_95high])
    
    columns = ['Control', 'Combination', 'Label', 'N_Control', 'N_Combination', 'Corr',
               'ITT_HR', 'ITT_HR_95low', 'ITT_HR_95high', 
               'Biomarker', 'Biomarker_HR', 'Biomarker_HR_95low', 'Biomarker_HR_95high']
    result = pd.DataFrame(data=data, columns=columns)
    return result.sort_values('Combination')


def create_monotherapy_input_sheet(indf: pd.DataFrame, config: dict) -> pd.DataFrame:
    # select only columns of interest
    indf = indf[indf['Figure'].isin(['additive', 'between'])].reset_index()
    df = indf[['Experimental', 'Control', 'Combination', 
               'Experimental First Scan Time', 'Control First Scan Time', 
               'Corr', 'N_experimental', 'N_control', 'N_combination', 'Figure']]
    
    # update to more recent data
    idx = df[df['Combination'] == 'Myeloma_Daratumumab-Carfilzomib+Dexamethasone_Dimopoulos2020_PFS'].index
    df.loc[idx, 'Combination'] = 'Myeloma_Daratumumab-Carfilzomib+Dexamethasone_Usmani2022_PFS'
    df.loc[idx, 'Control'] = 'Myeloma_Carfilzomib+Dexamethasone_Usmani2022_PFS'
    return df
    

def main():
    config = load_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, 
                        help='Dataset to use (main_combo, biomarker, single_agent)')
    args = parser.parse_args()

    if args.dataset == 'main_combo':
        master_df = pd.read_excel(config['data_master_sheet'], 
                                header=0, index_col=0, engine='openpyxl')
        master_df = master_df.dropna(subset=['Indication'])
        input_df = create_main_combo_input_sheet(master_df, config)

    elif args.dataset == 'biomarker':
        biomarker_sheet = pd.read_excel('data/clinical_trials/biomarker_data_sheet.xlsx',
                                        header=0, index_col=0, engine='openpyxl')
        input_df = create_biomarker_input_sheet(biomarker_sheet, config)

    elif args.dataset == 'single_agent':
        monotherapy_sheet = pd.read_csv("data/clinical_trials/single_agent/cox_ph_test.csv",
                                        header=0, index_col=None)
        input_df = create_monotherapy_input_sheet(monotherapy_sheet, config)
    
    else:
        raise ValueError(f'Dataset {args.dataset} is not supported.')
    
    input_df.to_csv(config[args.dataset]['metadata_sheet'], index=False)


if __name__ == '__main__':
    main()