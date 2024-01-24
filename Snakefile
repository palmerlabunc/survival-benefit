configfile: "config.yaml"

import pandas as pd

CONFIG_CELL = config["cell_line"]

rule all:
    input:
        # compute correlation
        f"{CONFIG_CELL['table_dir']}/CTRPv2_pairwise_{CONFIG_CELL['corr_method']}_correlation_distributions.csv",
        # biomarker analysis
        f"{config['biomarker']['fig_dir']}/actual_biomarker_vs_optimal_stratification.pdf",
        # single_agent analysis
        f"{config['single_agent']['fig_dir']}/compare_monotherapy_added_benefit_gini.pdf",
        # PDX analysis
        f"{config['PDX']['fig_dir']}/PDXE_combo_added_benefit_stripplot.pdf",
        # clinical data summary
        f'{config["main_combo"]["fig_dir"]}/therapy_modality_counts_barplot.pdf',
        # illustrative examples
        f"{config['example']['fig_dir']}/two_arms.pdf",
        # real example
        f"{config['example']['fig_dir']}/{config['example']['real_example_combo']}.two_arms.pdf",
        # main combo analysis
        f"{config['main_combo']['fig_dir']}/Surrogate.gini_compare_exp_and_high_kdeplot.pdf"


rule preprocess_experimental:
    input:
        f"{CONFIG_CELL['raw_dir']}/Table S2_Screened Cell Line Info_Ling et al_2018.xlsx"
        f"{CONFIG_CELL['raw_dir']}/Table S3_Screened Drug Info_Ling et al_2018.xlsx",
        f"{CONFIG_CELL['raw_dir']}/Gao2015_suppl_table.xlsx",
        "src/processing/process_CTRPv2_data.py",
        "src/processing/process_PDXE_data.py"
    output:
        f"{CONFIG_CELL['data_dir']}/CTRPv2_CCL.csv",
        f"{CONFIG_CELL['data_dir']}/CTRPv2_drug.csv",
        f"{CONFIG_CELL['data_dir']}/Recalculated_CTRP_12_21_2018.txt",
        f"{config['PDX']['data_dir']}/PDXE_model_info.csv",
        f"{config['PDX']['data_dir']}/PDXE_drug_response.csv"
    shell:
        "python src/processing/process_CTRPv2_data.py; "
        "python src/processing/process_PDXE_data.py"


rule precompute_correation:
    input:
        f"{CONFIG_CELL['data_dir']}/CTRPv2_CCL.csv",
        f"{CONFIG_CELL['data_dir']}/CTRPv2_drug.csv",
        f"{CONFIG_CELL['data_dir']}/Recalculated_CTRP_12_21_2018.txt",
        "src/processing/precompute_CTRPv2_correlation.py"
    output:
        f"{CONFIG_CELL['data_dir']}/PanCancer_all_pairwise_{CONFIG_CELL['corr_method']}_correlation.csv"
        # should have cancer-type specific all pairwise correlation
    shell:
        "python src/processing/precompute_CTRPv2_correlation.py"


rule compute_correlation:
    input:
        f"{CONFIG_CELL['data_dir']}/PanCancer_all_pairwise_{CONFIG_CELL['corr_method']}_correlation.csv",
        f"{config['PDX']['data_dir']}/PDXE_model_info.csv",
        f"{config['PDX']['data_dir']}/PDXE_drug_response.csv",
        "src/CTRPv2_correlation.py",
        "src/PDXE_correlation.py"
    output:
        f"{CONFIG_CELL['table_dir']}/CTRPv2_pairwise_{CONFIG_CELL['corr_method']}_correlation_distributions.csv",
        f"{config['PDX']['fig_dir']}/CM-BRAFmut_binimetinib_encorafenib.pdf",
        f"{config['PDX']['fig_dir']}/CRC-RASwt_cetuximab_5FU.pdf",
        f"{CONFIG_CELL['fig_dir']}/Colorectal_chemo_vs_targeted_corr_dist.pdf",
        f"{CONFIG_CELL['fig_dir']}/Breast_Lapatinib_vs_5-Fluorouracil.pdf"
        # should have cancer-type specific all pairwise correlation
    shell:
        "python src/CTRPv2_correlation.py; "
        "python src/PDXE_correlation.py"


rule create_input_sheets:
    input:
        f"{config['data_master_sheet']}",
        "data/clinical_trials/biomarker_data_sheet.xlsx",
        "data/clinical_trials/single_agent/cox_ph_test.csv",
        "src/create_input_sheets.py"
    output:
        #FIXME is there a way to retrieve the input sheet path from config?
        "data/clinical_trials/{dataset}_input_data_sheet.csv"
    shell:
        "python src/create_input_sheets.py {wildcards.dataset}"


rule clinical_data_summary:
    input:
        f"{config['data_master_sheet']}",
        "src/clinical_data_summary.py",
        "env/publication.mplstyle"
    output:
        f'{config["main_combo"]["table_dir"]}/therapy_modality_counts_barplot.source_data.csv',
        f'{config["main_combo"]["fig_dir"]}/therapy_modality_counts_barplot.pdf',
        f'{config["main_combo"]["table_dir"]}/cancer_type_counts_barplot.source_data.csv',
        f'{config["main_combo"]["table_dir"]}/correlation_distribution_histplot.source_data.csv',
        f'{config["main_combo"]["fig_dir"]}/correlation_distribution_histplot.pdf'
    shell:
        "python src/clinical_data_summary.py"


rule run_survival_benefit:
    input:
        #FIXME is there a way to retrieve the path from config?
        "data/clinical_trials/{dataset}_input_data_sheet.csv",
        "src/compute_benefit_all_combo.py",
        "src/survival_benefit/survival_benefit_class.py",
        "src/survival_benefit/survival_data_class.py",
        "env/publication.mplstyle"
    output:
        #FIXME is there a way to retrieve the path from config?
        directory("tables/{dataset}/predictions/")
    shell:
        "python src/compute_benefit_all_combo.py {wildcards.dataset}"


rule pdxe_anlaysis:
    input:
        f"{config['PDX']['data_dir']}/PDXE_drug_response.csv",
        "src/PDX_proof_of_concept.py",
        "src/PDX_proof_of_concept_helper.py",
        "src/survival_benefit/survival_benefit_class.py",
        "src/survival_benefit/survival_data_class.py",
        "env/publication.mplstyle",
        "src/utils.py"
    output:
        f"{config['PDX']['fig_dir']}/PDXE_combo_added_benefit_stripplot.pdf",
        f"{config['PDX']['table_dir']}/PDXE_combo_added_benefit_stripplot.source_data.csv",
        f"{config['PDX']['fig_dir']}/PDXE_added_benefit_distplot.pdf",
        f"{config['PDX']['fig_dir']}/PDXE_added_benefit_cumulative_distplot.pdf",
        f"{config['PDX']['fig_dir']}/PDXE_corr_differences_scatterplot.pdf",
        f"{config['PDX']['table_dir']}/PDXE_corr_differences_scatterplot.source_data.csv",
        f"{config['PDX']['fig_dir']}/PDXE_actual_vs_high_corr_benefit_profiles.pdf",
        f"{config['PDX']['fig_dir']}/PDXE_actual_vs_high_corr_{config['PDX']['delta_t']}_benefit_2lineplot.pdf",
        f"{config['PDX']['table_dir']}/PDXE_actual_vs_high_corr_{config['PDX']['delta_t']}_benefit_3lineplot.source_data.csv",
        f"{config['PDX']['fig_dir']}/PDXE_paired_test_for_antagonism_lineplot.pdf",
        f"{config['PDX']['table_dir']}/PDXE_paired_test_for_antagonism_lineplot.source_data.csv",
        f"{config['PDX']['fig_dir']}/PDXE_bootstrapping_test_for_antagonism.pdf",
    shell:
        "python src/PDX_proof_of_concept.py"


rule main_combo_analysis:
    input:
        f"{config['main_combo']['table_dir']}/predictions/",
        "src/main_combo_analysis.py",
        "src/survival_benefit/survival_benefit_class.py",
        "src/survival_benefit/survival_data_class.py",
        "env/publication.mplstyle",
        "src/utils.py"
    output:
        f"{config['main_combo']['fig_dir']}/Surrogate.gini_by_experimental_class_boxplot.pdf",
        f"{config['main_combo']['fig_dir']}/Surrogate.gini_histplot.pdf",
        f"{config['main_combo']['fig_dir']}/Surrogate.gini_compare_exp_and_high_kdeplot.pdf",
        f"{config['main_combo']['fig_dir']}/Surrogate.1mo_responder_percentage_barplot.pdf",
        f"{config['main_combo']['fig_dir']}/Surrogate.median_benefit_simulated_vs_actual_scatterplot_hightlight.pdf",
        f"{config['main_combo']['table_dir']}/exp_corr_stats_compiled.csv",
        f"{config['main_combo']['table_dir']}/high_corr_stats_compiled.csv"
    shell:
        "python src/main_combo_analysis.py"


rule compare_with_monotherapy:
    input:
        "config.yaml",
        f"{config['single_agent']['table_dir']}/predictions/",
        "src/compare_with_monotherapy.py",
        "src/survival_benefit/survival_benefit_class.py",
        "src/survival_benefit/survival_data_class.py",
        "env/publication.mplstyle",
        "src/utils.py"
    output:
        f"{config['single_agent']['table_dir']}/compare_monotherapy_added_benefit.csv",
        f"{config['single_agent']['fig_dir']}/compare_monotherapy_added_benefit_each_combo.pdf",
        f"{config['single_agent']['fig_dir']}/compare_monotherapy_added_benefit_rmse.pdf",
        f"{config['single_agent']['fig_dir']}/compare_monotherapy_added_benefit_gini.pdf",
    shell:
        "python src/compare_with_monotherapy.py"


rule biomarker_anlaysis:
    input:
        f"{config['main_combo']['table_dir']}/predictions/",
        f"{config['biomarker']['table_dir']}/predictions/",
        "src/biomarker_analysis.py",
        "src/survival_benefit/survival_benefit_class.py",
        "src/survival_benefit/survival_data_class.py",
        "env/publication.mplstyle"
    output:
        f"{config['biomarker']['table_dir']}/PFS.optimal_stratification_analysis.csv",
        f"{config['biomarker']['fig_dir']}/PFS.optimal_stratification_benefitter_scatterplot.pdf",
        f"{config['biomarker']['fig_dir']}/PFS.optimal_stratification_unselected_scatterplot.pdf",
        f"{config['biomarker']['table_dir']}/actual_biomarker_vs_optimal_stratification.csv",
        f"{config['biomarker']['fig_dir']}/actual_biomarker_vs_optimal_stratification.pdf",        
    shell:
        "python src/biomarker_analysis.py"


rule illustration:
    input:
        "config.yaml",
        "src/illustrative_example.py",
        "src/survival_benefit/survival_benefit_class.py",
        "src/survival_benefit/survival_data_class.py",
        "env/publication.mplstyle"
    output:
        expand("{fig_dir}/sorted_by_A_corr_{corr}_with_weibull.pdf", 
               fig_dir=config['example']['fig_dir'], corr=[0, 1]),
        expand("{fig_dir}/sorted_by_AB_corr_{corr}_with_weibull.pdf", 
               fig_dir=config['example']['fig_dir'], corr=[0, 1]),
        f"{config['example']['fig_dir']}/two_arms.pdf"
    shell:
        "python src/illustrative_example.py"


rule real_example:
    input:
        "config.yaml",
        "src/real_combo_example.py",
        "src/survival_benefit/survival_benefit_class.py",
        "src/survival_benefit/survival_data_class.py",
        "env/publication.mplstyle"
    output:
        directory(f"{config['example']['table_dir']}/predictions/"),
        f"{config['example']['fig_dir']}/{config['example']['real_example_combo']}.two_arms.pdf",
        f"{config['example']['fig_dir']}/{config['example']['real_example_experimental']}.pdf"
    shell:
        "python src/real_combo_example.py"