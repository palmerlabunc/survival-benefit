from datetime import datetime
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import spearmanr, rankdata
from survival_benefit.prob_functions import get_prob
from survival_benefit.survival_data_class import SurvivalData

COLORS = {'mono': '#00cd6c', 
          'comb': '#af58ba',
          'added_benefit': "#ffaa00",
          'low_bound': "#f28522"}


class SurvivalBenefit:
    """This version has the row approach and uses

    """
    def __init__(self,
                 mono_data: pd.DataFrame | SurvivalData = None,
                 comb_data: pd.DataFrame | SurvivalData = None,
                 mono_name: str = None, comb_name: str = None,
                 n_patients: int = 500,
                 atrisk: pd.DataFrame = None, corr='spearmanr',
                 outdir='.', out_name: str = None,
                 figsize=(6, 4), fig_format='png', save_mode=True):
        """
        Constructs SurvivalBenefit object. Valid combinations of arguments are:
        1. mono_/comb_data as a SurvivalData object
        2. mono_/comb_data as a pandas DataFrame with a name
        3. mono_/comb_name as a filepath prefix (before .csv)

        Args:
            mono_data (pd.DataFrame | SurvivalData, optional): monotherapy (control arm) data. Defaults to None.
            comb_data (pd.DataFrame | SurvivalData, optional): combination therapy data. Defaults to None.
            mono_name (str, optional): monotherapy name. 
                If a file path before the exension (.csv) is given, the csv file will be read. Defaults to None.
            comb_name (str, optional): combination therapy name. 
                If a file path before the exension (.csv) is given, the csv file will be read. Defaults to None.
            n_patients (int): Number of virtual patients to use for simulation. Defaults to 500.
            atrisk (pd.DataFrame, optional): at-risk table for monotherapy and control arms.
                If None, it will automatically look for the at-risk table from disk.  Defaults to None.
            corr (str, optional): correlation metrics (spearmanr or kendalltau). Defaults to 'spearmanr'.
            outdir (str, optional): output directory path. 
                SurvivalBenefit will create a subdirectory in `outdir` with the given `out_name`. Defaults to '.'.
                Will be ignored if `save_mode` is False.
            out_name (_type_, optional): output file prefix. If None, it will use combination therapy name. Defaults to None.
            figsize (tuple, optional): figure width and height. Defaults to (6, 4).
            fig_format (str, optional): figure format (png, pdf, or jpeg). Defaults to 'png'.
            save_mode (bool, optional): save the output to files. Defaults to True.
        """
        plt.style.use('survival_benefit.styles.publication')

        # initiate data attributes
        self.N = n_patients
        self.outdir = outdir
        self.atrisk = atrisk
        self.figsize = figsize
        self.fig_format = fig_format
        self.save_mode = save_mode

        if self.outdir.endswith('/'):
            self.outdir = self.outdir[:-1]

        # set at risk table
        self.atrisk = self.__set_atrisk_table(atrisk, comb_name)

        # CAUTION: Do not chnage orders of the functions below
        # sanity checks for data types for mono_data and comb_data
        if self.atrisk is None:
            self.mono_survival_data = self.__set_survival_data(
                mono_data, mono_name)
            self.comb_survival_data = self.__set_survival_data(
                comb_data, comb_name)

        else:
            self.mono_survival_data = self.__set_survival_data(
                mono_data, mono_name, atrisk=self.atrisk['control'])
            self.comb_survival_data = self.__set_survival_data(
                comb_data, comb_name, atrisk=self.atrisk['treatment'])

        self.__align_N()
        self.__align_tmax()
        self.__align_round()
        self.max_curve = self.__get_max_curve()
        self.norm_diff = self.__get_normalized_difference()
        self.rng = np.random.default_rng(0)

        # These will change with each run of compute_benefit
        self.info_str = ''
        self.benefit_df = None  # survival benefit profile dataframe
        self.corr_rho_target = None # target correlation
        self.corr_rho_actual = None # actual correlation
        self.used_bestmatch_corr = None
        self.prob_kind = None
        self.prob_coef = None
        self.prob_offset = None
        self.stats = pd.Series(np.nan,
                               index=['Monotherapy', 'Combination',
                                      'At_risk_table', 'Normalized_difference',
                                      'N',
                                      'Datetime',
                                      'Max_time',
                                      'Correlation_method',
                                      'Correlation_value_target',
                                      'Correlation_value_actual',
                                      'Correlation_pvalue',
                                      'Used_bestmatch_corr',
                                      'Prob_kind', 'Prob_coef', 'Prob_offset',
                                      'Gini_coefficient',
                                      'Median_benefit_lowbound', 
                                      'Median_benefit_highbound',
                                      'Percent_patients_valid',
                                      'Percent_patients_benefitting_1mo_from_total_highbound',
                                      'Percent_patients_benefitting_1mo_from_valid_highbound',
                                      'Percent_patients_benefitting_1mo_from_total_lowbound',
                                      'Percent_patients_benefitting_1mo_from_valid_lowbound',
                                      'Percent_patients_certain_from_total',
                                      'Percent_patients_leftbound_from_total'])
        
        # initialization of stats df
        
        self.stats['Monotherapy'] = self.mono_survival_data.name
        self.stats['Combination'] = self.comb_survival_data.name
        self.stats['N'] = self.N
        if self.atrisk is None:
            self.stats['At_risk_table'] = "No"
        else:
            self.stats['At_risk_table'] = "Yes"
        self.stats['Normalized_difference'] = np.round(self.norm_diff, 2)
        self.stats['Correlation_method'] = 'spearmanr'

        if self.save_mode:
            if out_name is None:
                self.outdir = self.__prepare_outdir(
                    self.comb_survival_data.name)
            else:
                self.outdir = self.__prepare_outdir(out_name)

    def __str__(self) -> str:
        return self.__generate_summary_stats_str()

    @staticmethod
    def acceptible_corr(corr: float, target_corr: float, diff_threshold: float = 0.005) -> bool:
        """Check if the correlation is acceptible.

        Args:
            corr (float): correlation
            target_corr (float): target correlation
            diff_threshold (float): difference threshold. Defaults to 0.005.

        Returns:
            bool: True if the correlation is acceptible
        """
        return abs(corr - target_corr) < diff_threshold

    @staticmethod
    def get_atrisk_filename_from_comb_name(comb_name: str) -> str:
        """Get the filename of the atrisk table from the combo survival data name.

        Args:
            comb_name (str): combo survival data name

        Returns:
            str: atrisk table filename
        """
        comb_file_tokens = comb_name.rsplit(
            '/', 1)  # split by last '/' to seperate directory path and file prefix
        comb_file_prefix = comb_file_tokens[-1]
        tokens = comb_file_prefix.split('_', 2)
        try:
            drug_names = tokens[1].split('-')
        except IndexError:
            return ""
        
        experimental_drug = drug_names[0]
        if len(comb_file_tokens) > 1:  # if it has a directory path
            atrisk_filepath = f'{comb_file_tokens[0]}/{tokens[0]}_{experimental_drug}_{tokens[2]}_at-risk.csv'
        else:
            atrisk_filepath = f'{tokens[0]}_{experimental_drug}_{tokens[2]}_at-risk.csv'
        return atrisk_filepath

    @staticmethod
    def compute_gini(values: np.ndarray, tmax: float = None) -> float:
        """Compute gini coefficient for given value. Values above tmax will be capped at tmax.

        Args:
            values (np.ndarray): array of all values (time)
            tmax (float): max value (time)

        Returns:
            float: gini coefficient
        """
        values = np.sort(values)
        # cap the values at tmax
        if tmax is not None:
            values[values > tmax] = tmax
        n = len(values)
        gini = (2 * np.sum(np.multiply(np.arange(1, n+1), values)) /
                (n * np.sum(values))) - (n + 1) / n
        return np.round(gini, 2)
    
    def set_tmax(self, tmax: float):
        self.tmax = tmax
        self.mono_survival_data.set_tmax(tmax)
        self.comb_survival_data.set_tmax(tmax)

    def compute_benefit(self, prob_coef=1.0, prob_offset=0.0, prob_kind='power'):
        """Compute benefit profile using the probability function.

        Args:
            prob_coef (float, optional): coefficent in probability function. Defaults to 1.0.
            prob_offset (float, optional): offset in probablity function. Defaults to 0.0.
            prob_kind (str, optional): how to calculate probability ('linear', 'power', 'exponential'). Defaults to 'power'.
        """
        self.benefit_df = self.__compute_benefit_helper(
            prob_coef, prob_offset, prob_kind)
        # record summary stats
        self.__record_summary_stats()

    def compute_benefit_at_corr(self, corr: float, use_bestmatch=False):
        """Compute the survival benefit of the added drug at a given correlation. The given correlation value
        will be rounded to two decimal points.

        Args:
            corr (float): correlation value. Must be between -1 and 1.
            use_bestmatch (bool, optional): use the best match correlation if the given correlation is not found. 
                Defaults to False.
        """
        assert corr >= -1 and corr <= 1, "Correlation must be between -1 and 1."
        corr = round(corr, 2)
        self.corr_rho_target = corr
        prob_coef_range = (-50, 50)
        benefit_df = self.__corr_search(corr, prob_coef_range, 0, 0.005)

        if benefit_df is not None:
            self.benefit_df = benefit_df            
            self.used_bestmatch_corr = False
            self.__record_summary_stats()
            return
        
        warnings.warn(f"Correlation {corr} not found.")
        if use_bestmatch:
            self.used_bestmatch_corr = True
            warnings.warn("Using best match correlation.")
            self.compute_benefit(-50)
            if self.corr_rho_target < self.corr_rho_actual:
                warnings.warn(f"Using lowest compatible correlation {self.corr_rho_actual} instead of given correlation {self.corr_rho_target}.")
            else:
                self.compute_benefit(50)
                warnings.warn(
                    f"Using highest compatible correlation {self.corr_rho_actual} instead of given correlation {self.corr_rho_target}.")

    def plot_compute_benefit_sanity_check(self, save=True, postfix="") -> (plt.Figure, plt.Axes):
        """Sanity check plot of monotherapy, updated, and combination arms.
        """
        plt.style.use('survival_benefit.styles.publication')
        if self.benefit_df is None:
            warnings.warn("Nothing to plot. Run compute_benefit first.")
            return

        mono_df = self.mono_survival_data.processed_data
        comb_df = self.comb_survival_data.processed_data

        try:
            mono_min_surv_idx = mono_df[mono_df['Time'] >= self.tmax].index[-1]
        except IndexError:
            mono_min_surv_idx = 0
        try:
            comb_min_surv_idx = comb_df[comb_df['Time'] >= self.tmax].index[-1]
        except IndexError:
            comb_min_surv_idx = 0
        # bounded by tmax
        left_bound_range = np.arange(
            mono_min_surv_idx, comb_min_surv_idx, 1) / self.N * 100

        sns.set_style('ticks')
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(mono_df.loc[mono_min_surv_idx:, 'Time'], mono_df.loc[mono_min_surv_idx:, 'Survival'],
                label='mono', color=COLORS['mono'])
        ax.plot(comb_df.loc[comb_min_surv_idx:, 'Time'], comb_df.loc[comb_min_surv_idx:, 'Survival'],
                label="comb", color=COLORS['comb'])
        tmp = self.benefit_df.sort_values('new_surv')
        ax.plot(tmp['new_t'], tmp['new_surv'],
                color=COLORS['added_benefit'], label="reconstructed")
        # defined region
        ax.fill_betweenx(np.arange(comb_min_surv_idx, self.N, 1) / self.N * 100,
                         mono_df.loc[comb_min_surv_idx:, 'Time'],
                         self.max_curve.loc[comb_min_surv_idx:, 'Time'],
                         color='orange', alpha=0.3)
        # low-bound region
        ax.fill_betweenx(left_bound_range,
                         mono_df.loc[mono_min_surv_idx:comb_min_surv_idx - 1, 'Time'],
                         np.repeat(self.tmax + 1, left_bound_range.size),
                         color='royalblue', alpha=0.3)
        # unknown region
        ax.axhspan(0, 100 * mono_min_surv_idx / self.N,
                   color='gray', alpha=0.3)  # gray shading for unknown
        ax.set_xlim(0, self.tmax + 1)
        ax.set_ylim(-1, 105)
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Patients (%)')
        ax.legend()
        sns.despine()
        if save and self.save_mode:
            fig.savefig(
                f'{self.outdir}/{self.info_str}{postfix}.sanity_check.{self.fig_format}', 
                bbox_inches='tight')

        return fig, ax

    def plot_t_delta_t_corr(self, save=True, postfix="") -> (plt.Figure, plt.Axes):
        """Plot survival time for A against added benefit from B.

        """
        plt.style.use('survival_benefit.styles.publication')
        if self.benefit_df is None:
            warnings.warn("Nothing to plot. Run compute_benefit first.")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        valid_subset = self.benefit_df[self.benefit_df['valid']]
        t_rank = rankdata(valid_subset['Time'], method='average')
        deltat_rank = rankdata(valid_subset['delta_t'], method='average')
        sns.scatterplot(x=t_rank,
                        y=deltat_rank,
                        hue=valid_subset['left_bound'],
                        hue_order=[0, 1],
                        palette=['orange', 'royalblue'],
                        ax=ax, alpha=0.5, s=10)
        sns.regplot(x=t_rank,
                    y=deltat_rank,
                    scatter=False, ax=ax, color='orange')

        ax.set_title(f'Spearmanr={self.corr_rho_actual:.2f}')
        sns.despine()
        ax.set_xlim(0, self.N + 10)
        ax.set_ylim(0, self.N + 10)
        ax.legend(title='low-bound', bbox_to_anchor=(1.05, 0.5))
        ax.set_xlabel('Rank by monotherapy survival time')
        ax.set_ylabel('Rank by delta_t survival time')
        if save and self.save_mode:
            fig.savefig(f'{self.outdir}/{self.info_str}{postfix}.corr.{self.fig_format}', 
                        bbox_inches='tight')

        return fig, ax

    def plot_gini_curve(self, save=True, postfix="") -> (plt.Figure, plt.Axes):
        """
        Plot gini curve.

        Args:
            save (bool, optional): _description_. Defaults to True.
            postfix (str, optional): _description_. Defaults to "".
        
        Returns:
            plt.Figure:
            plt.Axes:
        """
        plt.style.use('survival_benefit.styles.publication')
        df = self.benefit_df[self.benefit_df['valid']]
        cumsum = df['delta_t'].sort_values().cumsum().values
        norm_cumsum = 100 * cumsum / cumsum[-1]
        index = np.linspace(0, 100, len(norm_cumsum))

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(index, norm_cumsum)
        ax.set_xlabel('Patients (%)')
        ax.set_ylabel('Cumulative benefit (%)')
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 100)
        ax.plot([0, 100], [0, 100], linestyle='--', color='black')
        return ax
    
    def plot_benefit_distribution(self, save=True, postfix="", kind='absolute', simple=False) -> (plt.Figure, plt.Axes):
        plt.style.use('survival_benefit.styles.publication')
        if self.benefit_df is None:
            warnings.warn("Nothing to plot. Run compute_benefit first.")
            return
        
        if kind not in ['absolute', 'ratio']:
            raise ValueError("ERROR: Wrong kind parameter. It should be either 'absolute' or 'ratio'")
        
        
        if simple:
            return self.__plot_benefit_distribution_simple(save, postfix, kind)
        else:
            return self.__plot_benefit_distribution_extensive(save, postfix, kind)
        
    def __plot_benefit_distribution_simple(self, save=True, postfix="", kind='absolute') -> (plt.Figure, plt.Axes):
        """Plot simple version of the benefit distribution. Only plots in the valid region (as in 0-100% survival)
        and the (extrapolated) benefit.

        Args:
            save (bool, optional): _description_. Defaults to True.
            postfix (str, optional): _description_. Defaults to "".
            kind (str, optional): _description_. Defaults to 'absolute'.

        Returns:
            plt.Figure:
            plt.Axes:
        """
        valid_subset = self.benefit_df[self.benefit_df['valid']]
        stepsize = 100 / valid_subset.shape[0]
        if kind == 'absolute':
            delta = valid_subset['delta_t']  # defined
        elif kind == 'ratio':
            delta = 100 * valid_subset['delta_t'] / \
                valid_subset['Time']  # defined
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        delta_df = pd.DataFrame({'Survival': np.linspace(0, 100, valid_subset.shape[0]),
                                 'Time': np.sort(delta.values)[::-1]})

        ax.plot(delta_df['Time'], delta_df['Survival'],
                color=COLORS['added_benefit'])
        
        ax.fill_betweenx(delta_df['Survival'],
                         np.zeros(delta_df.shape[0]), delta_df['Time'],
                         color=COLORS['added_benefit'], alpha=0.3)

        ax.set_ylabel('Patients (%)')
        ax.set_ylim(-1, 105)

        if kind == 'absolute':
            ax.set_xlabel('Added benefit from B (months)')
            ax.set_xlim(0, self.tmax)

        elif kind == 'ratio':
            ax.set_xlabel('Fold increase in time (%)')
            ax.set_xlim(0, 500)

        if save and self.save_mode:
            fig.savefig(f'{self.outdir}/{self.info_str}{postfix}.distribution_simple_{kind}.{self.fig_format}', 
                        bbox_inches='tight')
        
        return fig, ax

    def __plot_benefit_distribution_extensive(self, save=True, postfix="", kind='absolute') -> (plt.Figure, plt.Axes):
        """Plot benefit distribution.

        """
        # set up data
        stepsize = 100 / self.N
        valid_subset = self.benefit_df[self.benefit_df['valid']]
        if kind == 'absolute':
            delta1 = valid_subset['lb_delta_t']  # low-bound
            delta2 = valid_subset['delta_t']  # defined
        elif kind == 'ratio':
            delta1 = 100 * valid_subset['lb_delta_t'] / \
                valid_subset['Time']  # low-bound
            delta2 = 100 * valid_subset['delta_t'] / \
                valid_subset['Time']  # defined

        unknown = 100 * (self.N - self.benefit_df['valid'].sum()) / self.N
        delta_df1 = pd.DataFrame({'Survival': np.linspace(unknown, 100 - stepsize, delta1.size),
                                  'Time': np.sort(delta1.values)[::-1]})
        delta_df2 = pd.DataFrame({'Survival': np.linspace(unknown, 100 - stepsize, delta2.size),
                                  'Time': np.sort(delta2.values)[::-1]})

        med_benefit_low = self.stats['Median_benefit_lowbound']
        med_benefit_high = self.stats['Median_benefit_highbound']

        # plot
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # low-bound
        ax.plot(delta_df1['Time'], delta_df1['Survival'],
                color=COLORS['low_bound'],
                label="low-bound benefit")
        ax.fill_betweenx(delta_df1['Survival'],
                         np.zeros(delta_df1.shape[0]), delta_df1['Time'],
                         color=COLORS['low_bound'], alpha=0.5)
        # defined
        ax.plot(delta_df2['Time'], delta_df2['Survival'],
                color=COLORS['added_benefit'],
                label="extrapolated benefit")
        ax.fill_betweenx(delta_df2['Survival'],
                         delta_df1['Time'], delta_df2['Time'],
                         color=COLORS['added_benefit'], alpha=0.3)

        # line dividng unknown and uncertain
        ax.axhline(unknown, linestyle='--', c='k')
        # gray shading for unknown
        ax.axhspan(0, unknown, color='gray', alpha=0.3)

        # add median range box
        if kind == 'absolute':
            ax.vlines(1, 0, 98, linestyle='--', color='k')  # 1mo mark
            # color background for median range
            ax.axhspan(100, 105, color='papayawhip', alpha=0.5)

            # lower bound annotation
            ax.annotate(str(round(med_benefit_low, 1)),
                        xy=(med_benefit_low, 100), xycoords='data',
                        xytext=(med_benefit_low - 2, 90), textcoords='data',
                        arrowprops=dict(arrowstyle="-", color='k'))
            # make median range rectangle
            med_range = patches.Rectangle((med_benefit_low, 100),
                                          med_benefit_high - med_benefit_low, 5,
                                          linewidth=1, color='black', alpha=0.5)
            # make upperbound annotation if not indefinite
            if med_benefit_high < self.tmax:

                ax.annotate(str(round(med_benefit_high, 1)),
                            xy=(med_benefit_high, 100), xycoords='data',
                            xytext=(med_benefit_high + 1, 90), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color='k'))
            # add "median" text
            if med_benefit_low > 5:
                ax.text(0.2, 100, "Median")
            else:
                ax.text(self.tmax - 5, 100, "Median")
            ax.add_patch(med_range)

            ax.set_xlabel('Added benefit from B (months)')
            ax.set_xlim(0, self.tmax)

        elif kind == 'ratio':
            ax.set_xlabel('Fold increase in time (%)')
            ax.set_xlim(0, 500)

        ax.set_ylabel('Patients (%)')
        ax.set_ylim(-1, 105)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.05), ncol=1,
                  borderaxespad=0, frameon=True)
        sns.despine()
        if save and self.save_mode:
            fig.savefig(
                f'{self.outdir}/{self.info_str}{postfix}.distribution_{kind}.{self.fig_format}', 
                bbox_inches='tight')
        
        return fig, ax

    def save_benefit_df(self, postfix=""):
        """Save benefit_df dataframe to csv file.

        """
        if self.benefit_df is None:
            warnings.warn("Nothing to save. Run compute_benefit first.")
            return
        
        assert self.save_mode, "Cannot Save Summary Stats: Not in Save Mode"
        self.benefit_df.round(2).to_csv(
            f'{self.outdir}/{self.info_str}{postfix}.table.csv')

    def save_summary_stats(self, postfix=""):
        """Save summary stats information.

        """
        if self.benefit_df is None:
            warnings.warn("Nothing to save. Run compute_benefit first.")
            return
        
        assert self.save_mode, "Cannot Save Summary Stats: Not in Save Mode"
        self.stats.to_csv(
            f'{self.outdir}/{self.info_str}{postfix}.summary_stats.csv',
            header=False)

    def __compute_benefit_helper(self, prob_coef, prob_offset=0.0, prob_kind='power') -> pd.DataFrame:
        """Internal helper function for computing the benefit profile.

        Args:
            prob_coef (float): coefficent in probability function
            prob_offset (float, optional): offset in probablity function. Defaults to 0.
            prob_kind (str, optional): how to calculate probability ('linear', 'power', 'exponential'). Defaults to 'power'.

        Returns:
            pd.DataFrame: survival benefit profile dataframe
        """
        assert self.norm_diff >= 0, f"Error: Cannot run algorithm for {self.comb_survival_data.name} if monotherapy is better than combination."
        self.prob_kind = prob_kind
        self.prob_coef = prob_coef
        self.prob_offset = prob_offset
        
        benefit_df = self.__initialize_benefit_df()
        mono_df = self.mono_survival_data.processed_data

        # start from the top (survival at 100%, shortest time)
        for i in range(self.N - 1, -1, -1):
            t = self.max_curve.at[i, 'Time']
            surv = self.max_curve.at[i, 'Survival']
            # from patients that have not been assigned 'benefit' yet
            patient_pool = benefit_df[~benefit_df['benefit']].index.values
            pool_size = patient_pool.size

            if pool_size == 0:  # if all patients have been assigned, continue
                continue
            elif pool_size == 1:
                chosen_patient = patient_pool[0]
            else:
                prob = get_prob(pool_size, prob_coef,
                                prob_offset, kind=prob_kind)
                chosen_patient = self.rng.choice(
                    patient_pool, 1, p=prob, replace=False)[0]

            t_chosen = mono_df.at[chosen_patient, 'Time']

            if t_chosen > t:
                try:
                    chosen_patient = benefit_df[(~benefit_df['benefit']) & (
                        benefit_df['Time'] <= t)].index.values[0]
                except IndexError:
                    continue

            benefit_df.at[chosen_patient, 'benefit'] = True
            benefit_df.at[chosen_patient, 'new_t'] = t
            benefit_df.at[chosen_patient, 'new_surv'] = surv

        # add delta_t info
        benefit_df.loc[:, 'delta_t'] = benefit_df['new_t'] - \
            benefit_df['Time']
        benefit_df.loc[:,
                       'delta_t'] = benefit_df['delta_t'].fillna(0)
        # low-bound delta t
        # This is when the assigned "new_t" is greater than tmax, i.e. in the weibull extrapolated region
        benefit_df.loc[benefit_df['new_t'] >= self.tmax, 'left_bound'] = True
        # in this case, lb_delta_t is the difference between tmax and monotherapy survival time
        benefit_df.loc[:, 'lb_delta_t'] = benefit_df['delta_t']
        benefit_df.loc[benefit_df['left_bound'], 'lb_delta_t'] = self.tmax - \
            benefit_df[benefit_df['left_bound']]['Time']

        return benefit_df

    def __compute_corr_from_benefit_df(self, benefit_df: pd.DataFrame) -> float:
        """Compute the correlation from the benefit_df.

        Args:
            benefit_df (pd.DataFrame): benefit dataframe

        Returns:
            float: correlation
            float: p-value
        """
        valid_subset = benefit_df[benefit_df['valid']]
        r2, p2 = spearmanr(valid_subset['Time'], valid_subset['delta_t'])
        return r2, p2

    def __corr_search_helper(self, prob_coef: float, prob_offset=0.0, prob_kind='power') -> float:
        """Helper function for corr_search.

        Args:
            prob_coef (float): probability coefficient
            prob_offset (float, optional): probability offset. Defaults to 0.0.
            prob_kind (str, optional): probability kind. Defaults to 'power'.

        Returns:
            pd.DataFrame: benefit dataframe
            float: actual correlation
        """
        benefit_df = self.__compute_benefit_helper(
            prob_coef, prob_offset, prob_kind)
        corr, p = self.__compute_corr_from_benefit_df(benefit_df)
        return benefit_df, corr

    def __corr_search(self, target_corr: float, prob_coef_range: tuple, prob_offset: float, diff_threshold: float):
        """Search for the probability coefficient and offset that gives the desired correlation.

        Args:
            target_corr (float): desired correlation
            prob_coef_range (tuple): probability coefficient range
            prob_offset (float): probability offset
            diff_threshold (float): difference threshold

        Returns:
            pd.DataFrame: benefit dataframe. None if not found.
        """
        coef_stepsize = 0.001

        # use standard binary search
        coef_low, coef_high = prob_coef_range
        benefit_df_low, corr_low = self.__corr_search_helper(
            coef_low, prob_offset, 'power')
        benefit_df_high, corr_high = self.__corr_search_helper(
            coef_high, prob_offset, 'power')
        if corr_low > target_corr or corr_high < target_corr:
            return None  # Not Found
        while corr_low <= corr_high:
            # don't forget to change this along with stepsize
            coef_mid = round((coef_low + coef_high) / 2, 3)
            benefit_df_mid, corr_mid = self.__corr_search_helper(
                coef_mid, prob_offset, 'power')
            if SurvivalBenefit.acceptible_corr(corr_mid, target_corr, diff_threshold):
                return benefit_df_mid
            if corr_mid < target_corr:
                coef_low = coef_mid + coef_stepsize
            else:
                coef_high = coef_mid - coef_stepsize
        return None  # Not Found

    def __record_summary_stats(self):
        """Record summary stats based on benefit_df. This method records the part where
        the values change with each run of compute_benefit.

        """
        if self.benefit_df is None:
            self.stats.iloc[5:] = None  # set to None from Datetime onwards
            return

        valid_subset = self.benefit_df[self.benefit_df['valid']]
        # benefitting percentage
        stepsize = 100 / self.N
        
        # percentage of patients with certain benefit (not at the wibull extrapolated region)
        percent_certain = stepsize * sum(~self.benefit_df['left_bound'] & self.benefit_df['valid'])
        
        percent_uncertain = stepsize * sum(self.benefit_df['left_bound'] & self.benefit_df['valid'])
        
        percent_1mo = stepsize * sum(valid_subset['delta_t'] >= 1)
        percent_1mo_lb = stepsize * sum(valid_subset['lb_delta_t'] >= 1)
        percent_valid = stepsize * valid_subset.shape[0]

        # median benefit among patients with >= 1 month low-bound benefit
        med_benefit_low = valid_subset[valid_subset['lb_delta_t'] >= 1]['lb_delta_t'].median()

        # median benefit among patients with >= 1 month benefit (weibull extrapolated)
        med_benefit_high = valid_subset[valid_subset['delta_t'] >= 1]['delta_t'].median()

        # correlation
        corr_rho_actual, corr_p = self.__compute_corr_from_benefit_df(self.benefit_df)
        self.corr_rho_actual = corr_rho_actual

        # record summary stats
        self.stats['Datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats['Max_time'] = np.round(self.tmax, 2)
        if self.corr_rho_target is not None:
            self.stats['Correlation_value_target'] = np.round(self.corr_rho_target, 2)
        self.stats['Correlation_value_actual'] = np.round(corr_rho_actual, 2)
        self.stats['Correlation_pvalue'] = corr_p
        self.stats['Used_bestmatch_corr'] = self.used_bestmatch_corr
        self.stats['Prob_kind'] = self.prob_kind
        self.stats['Prob_coef'] = self.prob_coef
        self.stats['Prob_offset'] = self.prob_offset
        self.stats['Gini_coefficient'] = self.compute_gini(
            valid_subset['delta_t'].values, self.tmax)
        self.stats['Median_benefit_lowbound'] = np.round(med_benefit_low, 2)
        self.stats['Median_benefit_highbound'] = np.round(med_benefit_high, 2)
        self.stats['Percent_patients_valid'] = np.round(percent_valid, 2)
        self.stats['Percent_patients_benefitting_1mo_from_total_highbound'] = np.round(percent_1mo, 2)
        self.stats['Percent_patients_benefitting_1mo_from_valid_highbound'] = np.round(100 * percent_1mo / percent_valid, 2)
        self.stats['Percent_patients_benefitting_1mo_from_total_lowbound'] = np.round(percent_1mo_lb, 2)
        self.stats['Percent_patients_benefitting_1mo_from_valid_lowbound'] = np.round(100 * percent_1mo_lb / percent_valid, 2)
        self.stats['Percent_patients_certain_from_total'] = np.round(percent_certain, 2)
        self.stats['Percent_patients_leftbound_from_total'] = np.round(percent_uncertain, 2)

        # generate info str
        self.info_str = self.__generate_info_str()
       
    def __set_survival_data(self, survival_data: SurvivalData | pd.DataFrame | None,
                            name: str | None,
                            atrisk: pd.Series = None) -> SurvivalData:
        """Sanity check the survival data types.

        Args:
            survival_data (SurvivalData | pd.DataFrame | None): survival data
            name (str | None): name or filepath to the survival data
            atrisk (pd.Series, optional): at risk table. Defaults to None.

        Raises:
            FileNotFoundError: if the survival data file is not found
            ValueError: if the survival data types are not the same

        Returns:
            SurvivalData: the survival data
        """
        # valid options
        # SurvivalData and no name provided
        if isinstance(survival_data, SurvivalData) and name is None:
            survival_data.add_weibull_tail()
            return survival_data
        # SurvivalData and name provided
        if isinstance(survival_data, SurvivalData) and name is not None:
            survival_data.add_weibull_tail()
            survival_data.set_name(name)
            return survival_data
        # pd.DataFrame and name provided
        if isinstance(survival_data, pd.DataFrame) and name is not None:
            return SurvivalData(name, survival_data, self.N, 
                                atrisk=atrisk, weibull_tail=True)
        # name is a filepath prefix (before .csv)
        if survival_data is None and name is not None:
            try:
                df = pd.read_csv(f'{name}.csv', index_col=None, header=0)
                real_name = name.split('/')[-1]
                return SurvivalData(real_name, df, self.N, 
                                    atrisk=atrisk, weibull_tail=True)
            except FileNotFoundError:
                raise FileNotFoundError(f"{name}.csv not found")
        # if does not match any valid options
        raise ValueError(
            "Invalid survival data type. Provide a SurvivalData object or a pandas DataFrame with a name.")

    def __set_atrisk_table(self, atrisk: pd.DataFrame | None, comb_name: str | None) -> pd.DataFrame | None:
        """Set the atrisk table. If atrisk is None, try to load the atrisk table from the comb_name.
        This function should be run before setting the SurvivalData.

        Args:
            atrisk (pd.DataFrame): atrisk table

        Returns:
            pd.DataFrame: the atrisk table. None if invalid.
        """
        if atrisk is None and comb_name is not None:
            try:
                atrisk_filepath = SurvivalBenefit.get_atrisk_filename_from_comb_name(
                    comb_name)
                atrisk = pd.read_csv(atrisk_filepath, index_col=None, header=0)
            except FileNotFoundError:
                warnings.warn(
                    f"at-risk table not found for {comb_name}. Setting atrisk table to None.")
                return None

        # check if the atrisk table is valid
        if not isinstance(atrisk, pd.DataFrame):
            return None
        elif not (atrisk.columns == ['control', 'treatment', 'time']).all():
            warnings.warn(
                "Invalid atrisk table. Must have columns ['control', 'treatment', 'time']. Setting atrisk table to None.")
            return None

        atrisk = atrisk.set_index('time')

        return atrisk

    def __align_tmax(self):
        """Align maximum time to be the minimum of the maximum follow-up times of the two curves.
        """
        tmax = min(self.mono_survival_data.tmax, self.comb_survival_data.tmax)
        self.set_tmax(tmax)

    def __align_N(self):
        """Match N of the combo and monotherapy survival data to be equal to the given N parameter.
        """
        if self.mono_survival_data.N != self.N:
            self.mono_survival_data.set_N(self.N)
        if self.comb_survival_data.N != self.N:
            self.comb_survival_data.set_N(self.N)

    def __align_round(self, decimals=5):
        self.mono_survival_data.round_time(decimals)
        self.comb_survival_data.round_time(decimals)

    def __prepare_outdir(self, name: str):
        """Create the output directory and return the output directory path

        Args:
            name (str): output directory name

        Returns:
            str: output directory full path
        """
        outdir = f'{self.outdir}/{name}'
        new_directory = Path(outdir)
        new_directory.mkdir(parents=True, exist_ok=True)
        return outdir

    def __get_normalized_difference(self):
        """ Compute the normalized difference in area between the two curves (in time scale)

        Returns:
            float: normalized difference
        """
        # calculate normalized difference between two curves
        comb_df = self.comb_survival_data.processed_data.copy()
        mono_df = self.mono_survival_data.processed_data.copy()

        # until tmax
        comb_df.loc[comb_df['Time'] > self.tmax, 'Time'] = self.tmax
        mono_df.loc[mono_df['Time'] > self.tmax, 'Time'] = self.tmax
        return 100 * (comb_df['Time'] - mono_df['Time']).sum() / self.N

    def __get_max_curve(self):
        """Trace the higher (maximum survival) parts of both curves. 

        Returns:
            pd.DataFrame: survival data frame
        """
        comb_df = self.comb_survival_data.processed_data
        mono_df = self.mono_survival_data.processed_data
        step = 100 / self.N
        max_curve = pd.DataFrame({'Time': np.maximum(mono_df['Time'], comb_df['Time']),
                                  'Survival': np.linspace(0, 100 - step, self.N)})
        return max_curve.round({'Time': 5,
                                'Survival': int(np.ceil(-np.log10(step)))})

    def __generate_info_str(self):
        if self.corr_rho_target is None:
            return f'N_{self.N}.target_corr_None.actual_corr_{self.corr_rho_actual:.2f}'
        else:
            return f'N_{self.N}.target_corr_{self.corr_rho_target:.2f}.actual_corr_{self.corr_rho_actual:.2f}'

    def __initialize_benefit_df(self):
        # initialize self.benefit_df
        benefit_df = self.mono_survival_data.processed_data.copy()
        benefit_df.index.name = 'patient_id'
        # did patient receive benefit from added drug?
        benefit_df.loc[:, 'benefit'] = False
        # only lower-bound of the benefit is determined
        benefit_df.loc[:, 'left_bound'] = False
        # monotherapy time >= max_time (i.e. invalid result, we don't know zone)
        benefit_df.loc[:, 'valid'] = True
        benefit_df.loc[benefit_df['Time'] >= self.tmax, 'valid'] = False

        benefit_df.loc[:, 'new_t'] = benefit_df['Time'].copy()
        benefit_df.loc[:, 'new_surv'] = benefit_df['Survival'].copy()
        benefit_df.loc[:, 'delta_t'] = np.nan
        benefit_df.loc[:, 'lb_delta_t'] = np.nan

        return benefit_df
