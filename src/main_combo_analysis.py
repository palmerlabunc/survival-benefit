from pathlib import Path
from multiprocessing import Pool
import shutil
import pandas as pd
import numpy as np
from survival_benefit.survival_benefit_class import SurvivalBenefit
from utils import load_config


def copy_if_not_exists(src_file: str, dest_file: str):
    """Copy a file from src to dest if dest does not exist.
    """
    dest_path = Path(dest_file)
    if not dest_path.exists():
        try:
            shutil.copy2(src_file, dest_file)
        except FileNotFoundError:
            print(f'File {src_file} does not exist.')


def copy_trial_files_to_data_dir(config: dict):
    """Copy the digitized KM curves and at-risk tables to the data directory.
    """
    data_dir = Path(config['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_sheet = pd.read_csv(config['metadata_sheet'], header=0, index_col=None)

    for _, row in metadata_sheet.iterrows():
        raw_dir = row['Raw Path']
        processed_dir = row['Processed Path']
        control_prefix = row['Control']
        combo_prefix = row['Combination']
        atrisk_filename = SurvivalBenefit.get_atrisk_filename_from_comb_name(combo_prefix)
        copy_if_not_exists(f'{raw_dir}/{control_prefix}.csv',
                           f'{processed_dir}/{control_prefix}.csv')
        copy_if_not_exists(f'{raw_dir}/{combo_prefix}.csv',
                           f'{processed_dir}/{combo_prefix}.csv')
        copy_if_not_exists(f'{raw_dir}/{atrisk_filename}',
                           f'{processed_dir}/{atrisk_filename}')


def compute_benefit_at_experimental_corr(data_dir: str, control_prefix: str, comb_prefix: str, 
                                         corr: float, out_dir: str, n: int = 500):
    """Compute the benefit at a given correlation.

    Args:
        data_dir (str): The directory of the data
        control_prefix (str): The prefix of the control file
        comb_prefix (str): The prefix of the combination file
        corr (float): The correlation to compute the benefit
        out_dir (str): The output directory
        n (int, optional): The number of virtual patients to use. Defaults to 500.
    """
    postfix_corr = str(round(corr, 2))[2:] # e.g. 0.34 -> 34
    survival_benefit = SurvivalBenefit(mono_name=f'{data_dir}/{control_prefix}', comb_name=f'{data_dir}/{comb_prefix}',
                                       n_patients=n, outdir=out_dir, figsize=(4, 3))
    
    survival_benefit.compute_benefit_at_corr(corr)
    survival_benefit.plot_compute_benefit_sanity_check(save=True, postfix=f'.corr_{postfix_corr}')
    survival_benefit.plot_t_delta_t_corr(
        save=True, postfix=f'.corr_{postfix_corr}')
    survival_benefit.plot_benefit_distribution(
        save=True, postfix=f'.corr_{postfix_corr}')
    survival_benefit.save_summary_stats(postfix=f'.corr_{postfix_corr}')
    survival_benefit.save_benefit_df(postfix=f'.corr_{postfix_corr}')


def compute_benefit_at_corr_for_all_main_combo(config: dict, n_process: int = 4):
    """Compute the benefit at a given correlation for all main combo in the input sheet.
    
    Args:
        config (dict): The config dictionary of the dataset
        n_process (int, optional): Number of processes to use. Defaults to 4.
    """
    input_sheet = pd.read_csv(
        config['metadata_sheet'], header=0, index_col=None)
    # create output directory
    out_dir = config['table_dir']
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # prepare arguments
    input_sheet.loc[:, 'Outdir'] = out_dir
    args_arr = input_sheet[['Processed Path', 'Control', 'Combination', 'Corr', 'Outdir']].values
    
    with Pool(processes=n_process) as pool:
        pool.starmap(compute_benefit_at_experimental_corr, args_arr)


def main():
    config = load_config()['main_combo']
    copy_trial_files_to_data_dir(config)
    compute_benefit_at_corr_for_all_main_combo(config, n_process=8)


if __name__ == '__main__':
    main()