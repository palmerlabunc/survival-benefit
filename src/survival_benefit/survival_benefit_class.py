from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import kendalltau, spearmanr, rankdata
from survival_benefit.prob_functions import get_prob
from survival_benefit.survival_data_class import SurvivalData


class SurvivalBenefit:
    """This version has the row approach and uses

    """

    def __init__(self, N: int, mono_data: SurvivalData, comb_data: SurvivalData,
                 atrisk=None, corr='spearmanr', 
                 outdir='.', out_name=None,
                 figsize=(6, 4), fig_format='png', save_mode=True):
        """
        Constructs SurvivalBenefit object.

        Args:
            N (int): Number of patients to use for simulation.
            mono_data (SurvivalData): monotherapy data
            comb_data (SurvivalData): combination therapy data
            atrisk (pd.DataFrame, optional): at-risk table for monotherapy and control arms. Defaults to None.
            corr (str, optional): correlation metrics (spearmanr or kendalltau). Defaults to 'spearmanr'.
            outdir (str, optional): output directory path. Defaults to '.'.
            out_name (_type_, optional): output file prefix. If None, it will use combination therapy name. Defaults to None.
            figsize (tuple, optional): figure width and height. Defaults to (6, 4).
            fig_format (str, optional): figure format (png, pdf, or jpeg). Defaults to 'png'.
            save_mode (bool, optional): _description_. Defaults to True.
        """        
        plt.style.use('survival_benefit.styles.publication')
        # initiate data attributes
        self.N = N
        self.outdir = outdir
        self.atrisk = atrisk
        self.figsize = figsize
        self.fig_format = fig_format
        self.save_mode = save_mode

        if self.outdir.endswith('/'):
            self.outdir = self.outdir[:-1]

        self.mono_survival_data = mono_data
        self.comb_survival_data = comb_data
        
        self.__align_N()
        self.__add_weibull_tails()
        self.__align_tmax()
        self.__align_round()
        self.max_curve = self.__get_max_curve()
        self.norm_diff = self.__get_normalized_difference()

        # These will change with each run of fit_curve
        self.benefit_df = None  # survival benefit profile dataframe
        self.info_str = None
        self.med_benefit_low = None  # median benefit of benefitting patients
        self.med_benefit_high = None
        self.percent_certain = None
        self.percent_uncertain = None  # left-bound
        self.percent_1mo = None  # percentage of patients with > 1 month benefit
        self.corr = corr  # correlation type: kendalltau or spearmanr
        self.corr_rho = None
        self.corr_p = None

        if self.save_mode:
            if out_name is None:
                self.outdir = self.__prepare_outdir(self.comb_survival_data.name)
            else:
                self.outdir = self.__prepare_outdir(out_name)
        


    def __str__(self) -> str:
        return self.__generate_summary_stats_str()
    
    def set_tmax(self, tmax):
        self.tmax = tmax
        self.mono_survival_data.set_tmax(tmax)
        self.comb_survival_data.set_tmax(tmax)


    def compute_benefit(self, prob_kind='power', prob_coef=1.0, prob_offset=0.0):
        """Compute the survival benefit of the added drug.

        Args:
            prob_kind (str, optional): how to calculate probability ('linear', 'power', 'exponential'). Defaults to 'power'.
            prob_coef (float, optional): coefficent in probability function. Defaults to 0.
            prob_offset (float, optional): offset in probablity function. Defaults to 0.
        """
        assert self.norm_diff >= 0, "Error: Cannot run algorithm if monotherapy is better than combination."

        rng = np.random.default_rng()

        self.info_str = self.__generate_info_str(
            self.N, prob_kind, prob_coef, prob_offset)

        self.benefit_df = self.__initialize_benefit_df()
        mono_df = self.mono_survival_data.processed_data

        for i in range(self.N - 1, -1, -1):
            t = self.max_curve.at[i, 'Time']
            surv = self.max_curve.at[i, 'Survival']
            # from patients that have not been assigned 'benefit' yet
            patient_pool = self.benefit_df[~self.benefit_df['benefit']].index.values
            pool_size = patient_pool.size

            if pool_size == 0:  # if all patients have been assigned, continue
                continue
            elif pool_size == 1:
                chosen_patient = patient_pool[0]
            else:
                prob = get_prob(pool_size, prob_coef,
                                prob_offset, kind=prob_kind)
                chosen_patient = rng.choice(
                    patient_pool, 1, p=prob, replace=False)[0]

            t_chosen = mono_df.at[chosen_patient, 'Time']

            if t_chosen > t:
                try:
                    chosen_patient = self.benefit_df[(~self.benefit_df['benefit']) & (
                        self.benefit_df['Time'] <= t)].index.values[0]
                except IndexError:
                    continue

            self.benefit_df.at[chosen_patient, 'benefit'] = True
            self.benefit_df.at[chosen_patient, 'new_t'] = t
            self.benefit_df.at[chosen_patient, 'new_surv'] = surv

        # add delta_t info
        self.benefit_df.loc[:, 'delta_t'] = self.benefit_df['new_t'] - \
            self.benefit_df['Time']
        self.benefit_df.loc[:,
                            'delta_t'] = self.benefit_df['delta_t'].fillna(0)
        # left-bound delta t
        self.benefit_df.loc[self.benefit_df['new_t']
                            >= self.tmax, 'left_bound'] = True
        self.benefit_df.loc[:, 'lb_delta_t'] = self.benefit_df['delta_t']
        self.benefit_df.loc[self.benefit_df['left_bound'], 'lb_delta_t'] = self.tmax - \
            self.benefit_df[self.benefit_df['left_bound']]['Time']
        # record summary stats
        self.__record_summary_stats()


    def __record_summary_stats(self):
        """Short summary.

        """
        valid_subset = self.benefit_df[self.benefit_df['valid']]
        # benefitting percentage
        stepsize = 100 / self.N
        self.percent_certain = stepsize * \
            sum((self.benefit_df['delta_t'] > 1) & (
                self.benefit_df['left_bound']) & (self.benefit_df['valid']))
        self.percent_uncertain = stepsize * \
            sum((self.benefit_df['lb_delta_t'] > 1) & (
                self.benefit_df['left_bound']) & (self.benefit_df['valid']))
        self.percent_1mo = stepsize * sum(valid_subset['delta_t'] > 1)

        # median benefit low bound
        self.med_benefit_low = valid_subset[valid_subset['lb_delta_t'] > 1]['lb_delta_t'].median(
        )

        # median benefit high bound
        self.med_benefit_high = valid_subset[valid_subset['delta_t'] > 1]['delta_t'].median(
        )

        # correlation
        if self.corr == 'kendalltau':
            r2, p2 = kendalltau(valid_subset['Time'], valid_subset['delta_t'])
        elif self.corr == 'spearmanr':
            r2, p2 = spearmanr(valid_subset['Time'], valid_subset['delta_t'])
        self.corr_rho = r2
        self.corr_p = p2


    def plot_compute_benefit_sanity_check(self, save=True, postfix=""):
        """Sanity check plot of monotherapy, updated, and combination arms.
        """

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
        left_bound_range = np.arange(
            mono_min_surv_idx, comb_min_surv_idx, 1) / self.N * 100

        sns.set_style('ticks')
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(mono_df.loc[mono_min_surv_idx:, 'Time'], mono_df.loc[mono_min_surv_idx:, 'Survival'],
                label='mono', color='green')
        ax.plot(comb_df.loc[comb_min_surv_idx:, 'Time'], comb_df.loc[comb_min_surv_idx:, 'Survival'],
                label="comb", color='purple')
        tmp = self.benefit_df.sort_values('new_surv')
        ax.plot(tmp['new_t'], tmp['new_surv'],
                color='orange', label="reconstructed")
        # defined region
        ax.fill_betweenx(np.arange(comb_min_surv_idx, self.N, 1) / self.N * 100,
                         mono_df.loc[comb_min_surv_idx:, 'Time'],
                         self.max_curve.loc[comb_min_surv_idx:, 'Time'],
                         color='orange', alpha=0.3)
        # left-bound region
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
        ax.set_ylabel('Probability (%)')
        ax.legend()
        sns.despine()
        if save and self.save_mode:
            fig.savefig(
                f'{self.outdir}/{self.info_str}{postfix}.sanity_check.{self.fig_format}', bbox_inches='tight')
            plt.close()

        return ax


    def plot_t_delta_t_corr(self, save=True, postfix=""):
        """Plot survival time for A against added benefit from B.

        """
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

        ax.set_title('{corr}={r:.2f} p={p:.2e}'.format(
            corr=self.corr, r=self.corr_rho, p=self.corr_p))
        sns.despine()
        ax.set_xlim(0, self.N + 10)
        ax.set_ylim(0, self.N + 10)
        ax.legend(title='left-bound', bbox_to_anchor=(1.05, 0.5))
        ax.set_xlabel('Rank by monotherapy survival time')
        ax.set_ylabel('Rank by delta_t survival time')
        if save and self.save_mode:
            fig.savefig(
                f'{self.outdir}/{self.info_str}{postfix}.corr.{self.fig_format}', bbox_inches='tight')
            plt.close()

        return ax

    def plot_benefit_distribution(self, save=True, postfix="", kind='absolute'):
        """Plot benefit distribution.

        """
        # set up data
        sns.set_style('ticks')
        stepsize = 100 / self.N
        valid_subset = self.benefit_df[self.benefit_df['valid']]
        if kind == 'absolute':
            delta1 = valid_subset['lb_delta_t']  # left-bound
            delta2 = valid_subset['delta_t']  # defined
        elif kind == 'ratio':
            delta1 = 100 * valid_subset['lb_delta_t'] / \
                valid_subset['Time']  # left-bound
            delta2 = 100 * valid_subset['delta_t'] / \
                valid_subset['Time']  # defined
        else:
            print("ERROR: Wrong kind parameter", file=sys.stderr)
            return
        unknown = 100 * (self.N - self.benefit_df['valid'].sum()) / self.N
        delta_df1 = pd.DataFrame({'Survival': np.linspace(unknown, 100 - stepsize, delta1.size),
                                  'Time': np.sort(delta1.values)[::-1]})
        delta_df2 = pd.DataFrame({'Survival': np.linspace(unknown, 100 - stepsize, delta2.size),
                                  'Time': np.sort(delta2.values)[::-1]})

        # plot
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # left-bound
        ax.plot(delta_df1['Time'], delta_df1['Survival'],
                color='orange',
                label="left-bound benefit")
        ax.fill_betweenx(delta_df1['Survival'],
                         np.zeros(delta_df1.shape[0]), delta_df1['Time'],
                         color='orange', alpha=0.5)
        # defined
        ax.plot(delta_df2['Time'], delta_df2['Survival'],
                color='royalblue',
                label="extrapolated benefit")
        ax.fill_betweenx(delta_df2['Survival'],
                         delta_df1['Time'], delta_df2['Time'],
                         color='royalblue', alpha=0.3)

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
            ax.annotate(str(round(self.med_benefit_low, 1)),
                        xy=(self.med_benefit_low, 100), xycoords='data',
                        xytext=(self.med_benefit_low - 2, 90), textcoords='data',
                        arrowprops=dict(arrowstyle="-", color='k'))
            # make median range rectangle
            med_range = patches.Rectangle((self.med_benefit_low, 100),
                                          self.med_benefit_high - self.med_benefit_low, 5,
                                          linewidth=1, color='black', alpha=0.5)
            # make upperbound annotation if not indefinite
            if self.med_benefit_high < self.tmax:

                ax.annotate(str(round(self.med_benefit_high, 1)),
                            xy=(self.med_benefit_high, 100), xycoords='data',
                            xytext=(self.med_benefit_high + 1, 90), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color='k'))
            # add "median" text
            if self.med_benefit_low > 5:
                ax.text(0.2, 100, "Median")
            else:
                ax.text(self.tmax - 5, 100, "Median")
            ax.add_patch(med_range)

            ax.set_xlabel('Added benefit from B (months)')
            ax.set_xlim(0, self.tmax)

        elif kind == 'ratio':
            ax.set_xlabel('Fold increase in time (%)')
            ax.set_xlim(0, 500)

        ax.set_ylabel('Probability (%)')
        ax.set_ylim(-1, 105)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.05), ncol=1,
                  borderaxespad=0, frameon=True)
        sns.despine()
        if save and self.save_mode:
            fig.savefig(
                f'{self.outdir}/{self.info_str}{postfix}.distribution_{kind}.{self.fig_format}', bbox_inches='tight')
            plt.close()
        return ax


    def save_benefit_df(self, postfix=""):
        """Save benefit_df dataframe to csv file.

        """
        assert self.save_mode, "Cannot Save Summary Stats: Not in Save Mode"
        self.benefit_df.to_csv(
            f'{self.outdir}/{self.info_str}{postfix}.table.csv')


    def save_summary_stats(self, postfix=""):
        """Save summary stats information.

        """
        assert self.save_mode, "Cannot Save Summary Stats: Not in Save Mode"
        summary = self.__generate_summary_stats_str()

        with open(f'{self.outdir}/{self.info_str}{postfix}.summary_stats.tsv', 'w') as f:
            f.write(summary)


    def __add_weibull_tails(self):
        """Add weibull tails to the mono and combo survival data if they don't exist
        """
        # add weibull tails to the survival data
        self.mono_survival_data.add_weibull_tail()
        self.comb_survival_data.add_weibull_tail()


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
        comb_df = self.comb_survival_data.processed_data
        mono_df = self.mono_survival_data.processed_data
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
    

    def __generate_info_str(self, n: int, prob_kind: str, prob_coef: float, prob_offset: float):
        return f'N_{n}_prob_{prob_kind}_{prob_coef}_{prob_offset}'


    def __generate_summary_stats_str(self):
        if self.atrisk is None:
            atrisk_status = "No"
        else:
            atrisk_status = "Yes"
        
        textstr = '\n'.join((f"Date\t{date.today()}",
                             f"Info\t{self.info_str}",
                             f"Monotherapy\t{self.mono_survival_data.name}",
                             f"Combination\t{self.comb_survival_data.name}",
                             f"At_risk_table\t{atrisk_status}",
                             "Normalized_difference\t{:.2f}".format(
                                 self.norm_diff),
                             "Median_benefit_lowbound\t{:.2f}".format(
                                 self.med_benefit_low),
                             "Median_benefit_highbound\t{:.2f}".format(
                                 self.med_benefit_high),
                             "Percentage_of_patients_benefitting_1mo\t{:.2f}".format(
                                 self.percent_1mo),
                             "Percentage_of_patients_benefitting_certain\t{:.2f}".format(
                                 self.percent_certain),
                             "Percentage_of_patients_benefitting_leftbound\t{:.2f}".format(
                                 self.percent_uncertain),
                             "Percentage_of_patients_valid\t{:.2f}".format(
                                 100 * np.sum(self.benefit_df['valid']) / self.N),
                             "Max_time\t{:.2f}".format(self.tmax),
                             "{0}\t{1:.2f}".format(
                                 self.corr, self.corr_rho),
                             "{0}_pvalue\t{1:.2e}".format(self.corr, self.corr_p)))
        return textstr
    
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

    
