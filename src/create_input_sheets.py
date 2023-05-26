from utils import load_config
import pandas as pd

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
            print(i)
        n_combo = int(row['Combination Arm N'])
        n_control = int(row['Control Arm N'])
        corr = row['Pearsonr']

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


def create_biomarker_input_sheet(master_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...


def create_monotherapy_input_sheet(master_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...


def create_placebo_input_sheet(master_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...


def main():
    config = load_config()
    master_df = pd.read_excel(config['data_master_sheet'], 
                              header=0, index_col=0, engine='openpyxl')
    master_df = master_df.dropna(subset=['Indication'])
    main_df = create_main_combo_input_sheet(master_df, config)

    #biomarker_df = create_biomarker_input_sheet(master_df, config)
    #monotherapy_df = create_monotherapy_input_sheet(master_df, config)
    #placebo_df = create_placebo_input_sheet(master_df, config)

    main_df.to_csv(config['main_combo']['metadata_sheet'], index=False)
    #biomarker_df.to_csv(config['biomarker']['metadata_sheet'], index=False)
    #monotherapy_df.to_csv(config['single_agent']['metadata_sheet'], index=False)
    #placebo_df.to_csv(config['placebo']['metadata_sheet'], index=False)


if __name__ == '__main__':
    main()