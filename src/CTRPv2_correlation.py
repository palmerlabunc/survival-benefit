#!/usr/bin/env python

import pandas as pd
import numpy as np
from itertools import combinations_with_replacement
from collections.abc import Iterable
from processing.precompute_CTRPv2_correlation import import_ctrp_data, precompute_moa_drug_list
from utils import load_config

_, DRUG_INFO, _, _ = import_ctrp_data()
CHEMO_DRUGS, TARGETED_DRUGS = precompute_moa_drug_list(DRUG_INFO)


def get_moa_drug_list(drug_list: Iterable[str], setname: str, chemo_drug_list=None, targeted_drug_list=None) -> np.array:
    """Get drugs with setname MoA
    
    Args:
        drug_list (Iterable[str]): list of drugs
        setname (str): chemo or targeted

    Returns:
        np.array: drug list with given setname MoA.
    """
    assert setname in ['all', 'chemo', 'targeted']
    
    if setname == 'all':
        return drug_list

    if setname == 'chemo':
        if chemo_drug_list is None:
            _, drug_info, _, _ = import_ctrp_data()
            drugs_with_moa, _ = precompute_moa_drug_list(drug_info)
        else:
            drugs_with_moa = chemo_drug_list
    elif setname == 'targeted':
        if targeted_drug_list is None:
            _, drug_info, _, _ = import_ctrp_data()   
            _, drugs_with_moa = precompute_moa_drug_list(drug_info)
        else:
            drugs_with_moa = targeted_drug_list

    return np.intersect1d(drug_list, drugs_with_moa)
    

def prepare_ctrp_agg_data(data_dir: str, cancer_type='PanCancer', drugA_setname="all", drugB_setname="all") -> np.ndarray:
    """Returns an array of correlation values of given drug pairs in the given cancer type

    Args:
        data_dir (str): directory where the precomputed correlation tables are stored.
        cancer_type (str, optional): Cancer type of interest. Defaults to 'PanCancer'.
        drugA_setname (str, optional): 'all', 'chemo', 'targeted', or specific drug. Defaults to "all".
        drugB_setname (str, optional): 'all', 'chemo', or 'targeted'. Defaults to "all".

    Returns:
        np.ndarray: array of correlation values for given pairs
    """

    try:
        df = pd.read_csv(f'{data_dir}/{cancer_type}_all_pairwise_correlation.csv',
                        index_col=0, header=0)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{cancer_type} cancer type not available. Please use PanCacner instead.")
    
    assert (drugA_setname in ['all', 'chemo', 'targeted']) or (drugA_setname in df.index)
    assert drugB_setname in ['all', 'chemo', 'targeted']
    df.astype(np.float64)
    np.fill_diagonal(df.values, np.nan)  # convert diagonal values to NaN (correlation with itself)

    # all vs. all -> return immediately
    if drugA_setname == 'all' and drugB_setname == 'all':
        vals = df.values.flatten()
        vals = vals[~np.isnan(vals)]
        return vals

    set_vs_set_flag = (drugA_setname in ['all', 'chemo', 'targeted']) and (
        drugB_setname in ['all', 'chemo', 'targeted'])
    
    # set vs. set
    if set_vs_set_flag:
        drugA_list = get_moa_drug_list(df.index, drugA_setname, chemo_drug_list=CHEMO_DRUGS, targeted_drug_list=TARGETED_DRUGS)
        drugB_list = get_moa_drug_list(
            df.index, drugB_setname, chemo_drug_list=CHEMO_DRUGS, targeted_drug_list=TARGETED_DRUGS)

        vals = df.loc[drugA_list, drugB_list].values.flatten()
        vals = vals[~np.isnan(vals)]
        return vals
    
    # one vs. set
    drugB_list = get_moa_drug_list(df.index, drugB_setname)
    vals = df.loc[drugA_setname, drugB_list].values
    vals = vals[~np.isnan(vals)]
    return vals


def calc_corr_dist_summary_stats(pairs: np.array) -> tuple[float, float, float, float, float]:
    """Calaculate summary statstics (sample size, mean, std, quantiles) of given
    correlation values.

    Args:
        pairs (np.array): 1D array of given correlation values

    Returns:
        float: sample size
        float: mean of given correlation values
        float: std of given correlation values
        float: lower 95% of given correlation values
        float: upper 95% of given correlation values
    """
    n = len(pairs)
    avg = round(np.mean(pairs), 5)
    std = round(np.std(pairs), 5)
    lower = round(np.quantile(pairs, 0.025), 5)
    upper = round(np.quantile(pairs, 0.975), 5)

    return n, avg, std, lower, upper


def get_CTRPv2_one_vs_one_correlation(data_dir: str, cancer_type: str, drugA: str, drugB: str) -> float:
    """Get one drug vs. one drug correlation value from precomputed correlation data for CTRPv2

    Args:
        data_dir (str): directory where the precomputed correlation tables are stored.
        cancer_type (str): Cancer type of interest
        drugA (str): Harmonized Name of drug A (must be in CTRPv2 data)
        drugB (str): Harmonized Name of drug A (must be in CTRPv2 data)

    Raises:
        FileNotFoundError: if correlation for the given cancer type could not be found

    Returns:
        float: pearsonr correlation coefficient
    """
    try:
        df = pd.read_csv(f'{data_dir}/{cancer_type}_all_pairwise_correlation.csv',
                        index_col=0, header=0)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{cancer_type} cancer type not available. Please use PanCacner instead.")
    
    return round(df.loc[drugA, drugB], 5)


def report_CTRPv2_correlation(data_dir: str, outfile: str):
    cancer_types = ['PanCancer', 'Breast', 'Cervical', 'Colorectal', 'Gastric', 'HeadNeck', 'Leukemia',
                    'Lung', 'Lymphoma', 'Melanoma', 'Myeloma', 'Ovarian', 'Pancreatic', 'Renal']
    moas = ['all', 'targeted', 'chemo']
    one_vs_sets = [('Breast', 'Lapatinib', 'all'), 
                     ('Colorectal', 'Topotecan', 'targeted'),
                     ('Gastric', 'Paclitaxel', 'chemo'),
                     ('Gastric', 'Paclitaxel', 'targeted'),
                     ('PanCancer', 'Docetaxel', 'chemo'),
                     ('Leukemia', 'ibrutinib', 'all'),
                     ('Leukemia', 'Idelalisib', 'targeted'),
                     ('Lung', 'Gemcitabine', 'chemo'),
                     ('Lung', 'Docetaxel', 'targeted'),
                     ('Lung', 'Erlotinib', 'targeted'),
                     ('Lung', 'Vincristine sulfate', 'chemo'),
                     ('Ovarian', 'Gemcitabine', 'chemo')]
    
    one_vs_one = [('Breast', 'Gemcitabine', 'Paclitaxel'),
                  ('Breast', 'Lapatinib', '5-Fluorouracil'),
                  ('PanCancer', 'Topotecan', 'Oxaliplatin'),
                  ('Leukemia', 'Navitoclax', 'Decitabine'),
                  ('PanCancer', 'Methotrexate', 'Oxaliplatin'),
                  ('Myeloma', 'Doxorubicin', 'Bortezomib'),
                  ('Pancreatic', 'Erlotinib', 'Gemcitabine'),
                  ('Pancreatic', 'Topotecan', '5-Fluorouracil'),
                  ('Pancreatic', 'Paclitaxel', 'Gemcitabine')]
    
    with open(outfile, 'w') as fout:
        fout.write('Cancer_type,drugA_set,drugB_set,n,mean,std,lower2.5%,upper97.5%\n')
        for cancer in cancer_types:
            for x, y in combinations_with_replacement(moas, 2):
                try:
                    pairwise_corr = prepare_ctrp_agg_data(data_dir, cancer, drugA_setname=x, drugB_setname=y)
                    l, mean, std, lower, upper = calc_corr_dist_summary_stats(pairwise_corr)
                    fout.write(f'{cancer},{x},{y},{l},{mean},{std},{lower},{upper}\n')
                except FileNotFoundError:
                    pass
        for cancer, x, y in one_vs_sets:
            try:
                pairwise_corr = prepare_ctrp_agg_data(data_dir, cancer, drugA_setname=x, drugB_setname=y)
                l, mean, std, lower, upper = calc_corr_dist_summary_stats(pairwise_corr)
                fout.write(f'{cancer},{x},{y},{l},{mean},{std},{lower},{upper}\n')
            except FileNotFoundError:
                pass
            except AssertionError:
                print(cancer, x, y)
        for cancer, x, y in one_vs_one:
            r = get_CTRPv2_one_vs_one_correlation(data_dir, cancer, x, y)
            fout.write(f'{cancer},{x},{y},1,{r},NA,NA,NA\n')


def main():
    config = load_config()

    data_dir = config['cell_line']['data_dir']
    table_dir = config['cell_line']['table_dir']

    report_CTRPv2_correlation(data_dir, f'{table_dir}/CTRPv2_pairwise_correlation_distributions.csv')


if __name__ == '__main__':
    main()
