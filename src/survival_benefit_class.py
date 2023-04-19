from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import kendalltau, spearmanr, rankdata
from prob_functions import get_prob
from processing.cleanup_KMcurves import preprocess_survival_data, cleanup_survival_data
from utils import interpolate, weibull_from_digitized


class SurvivalBenefit:
    """This version has the row approach and uses

    """

    def __init__(self, N: int, mono_name: str, comb_name: str,
                 mono_data=None, comb_data=None, atrisk=None,
                 indir=None, outdir='.', out_name=None,
                 figsize=(6, 4), corr='spearmanr', fig_format='png', save_mode=True):
        """ Constructs SurvivalBenefit object. Input data should be given by specifying
        the input directory (will read files that has mono_name and comb_name)
        or giving dataframes for mono and combination therapy data.
        If both are given, dataframe method will override.

        Parameters
        ----------
        N : int
            Number of patients to use for simulation.
        mono_name : str
            Name or file prefix of monotherapy data.
        comb_name : str
            Name or file prefix of combination therapy data.
        mono_data : pandas.DataFrame
            Monotherapy data containing time and survival. (default: None)
        comb_data : pandas.DataFrame
            Combination therapy data containing time and survival.
            (default: None)
        atrisk : str
            Atrisk table filename prefix. (the default is None.) If not provided,
            it will search for a file with the following naming convention:
            {CancerType}_{AddedDrug}_{DataSource}_{Metric}_at-risk.csv.
            atrisk table should have three columns: control, treatment, time
        indir : str
            Input file directory path (the default is None).
        outdir : str
            Outpt directory path.
        out_name : str
            Output file prefix. If None, it will use `comb_name` (the default is None).
        figsize : tuple
            Output figure size in width and height (the default is (6, 4)).
        corr : str
            Correlation metrics choices of spearmanr and kendalltau (the default is 'spearmanr').
        fig_format : str
            Figure format choices of png, pdf, and jpeg (the default is 'png').

        """

        # initiate data attributes
        self.N = N
        self.indir = indir
        self.outdir = outdir
        self.figsize = figsize
        self.fig_format = fig_format
        self.save_mode = save_mode
        self.out_name = out_name
        self.mono_name = mono_name
        self.comb_name = comb_name

        # set indir
        if (mono_data is None) and (comb_data is None):
            if self.indir.endswith('/'):
                self.indir = self.indir[:-1]
        
        if self.outdir.endswith('/'):
            self.outdir = self.outdir[:-1]

        # atrisk-table
        self.atrisk = self.__get_atrisk_table(atrisk)

        # prepare mono and comb data
        assert (indir is not None) or ((mono_data is not None)
                                       and (comb_data is not None)), "No input data provided"
        if (mono_data is None) and (comb_data is None):  # import from data file
            try:
                self.ori_mono = pd.read_csv(f'{self.indir}/{mono_name}.clean.csv')
                self.ori_comb = pd.read_csv(f'{self.indir}/{comb_name}.clean.csv')
            except FileNotFoundError:
                self.ori_mono = preprocess_survival_data(f'{self.indir}/{mono_name}.clean.csv')
                self.ori_comb = preprocess_survival_data(f'{self.indir}/{comb_name}.clean.csv')

        else:  # import from data frame
            self.ori_mono = self.__import_survival_data_from_pandas(mono_data)
            self.ori_comb = self.__import_survival_data_from_pandas(comb_data)

        self.mono = None
        self.comb = None
        self.time_max = min(self.ori_mono['Time'].max(), self.ori_comb['Time'].max())
        self.max_curve = None
        self.w_mono = None
        self.w_comb = None
        self.norm_diff = None  # normalized difference between two curves
        self.prepare_dataframe(self.time_max)

        # These will change with each run of fit_curve
        self.record = None  # record dataframe
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
                self.outdir = self.__prepare_outdir(self.comb_name)
            else:
                self.outdir = self.__prepare_outdir(out_name)

    def __get_atrisk_table(self, filename):
        # first look for given file name, if exists
        if filename is not None:
            try:
                atrisk = pd.read_csv(f'{self.indir}/{filename}.csv',
                                     header=0, index_col=None)
            except FileNotFoundError:
                pass

        # look for conventional name
        # 0: cancer type / 1: drugs / 2: data source
        # 3: survival metrics / 4:~ additional
        tokens = self.comb_name.split('_')
        try:
            drug = tokens[1].split('-')[0]
            filename = "_".join(
                (tokens[0], drug, tokens[2], tokens[3], 'at-risk'))
        except IndexError:
            filename = "ShouldNotExist"
        try:
            atrisk = pd.read_csv(f'{self.indir}/{filename}.csv',
                                 header=0, index_col='time', dtype=float)
        except (TypeError, FileNotFoundError) as e:
            atrisk = None
        return atrisk


    def __import_survival_data_from_pandas(self, data):
        """Import survival data from data frame having two columns
        (time, survival) or one column (time).

        Parameters
        ----------
        data : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            Survival Data.

        """
        df = data.copy()
        if df.shape[1] == 1:
            df.columns = ['Time']
            df = df.sort_values(['Time'], ascending=False).reset_index()
            df.loc[:, 'Survival'] = np.linspace(0, 100, num=df.shape[0])
        else:
            df.columns = ['Time', 'Survival']  # change column names
        # convert to percentage
        if df['Survival'].max() <= 1.1:
            df.loc[:, 'Survival'] = df['Survival'] * 100

        df = cleanup_survival_data(df)
        return df


    def __get_weibull_fit(self):
        """Create weibull fitted survival data for correlation calculation.

        Returns
        -------
        tuple
            tuple of pandas.DataFrame each for mono and comb

        """
        w_mono = weibull_from_digitized(self.mono, self.N, self.time_max)
        w_comb = weibull_from_digitized(self.comb, self.N, self.time_max)

        w_max = min(w_mono['Time'].max(), w_comb['Time'].max())
        # align max time
        w_mono.loc[w_mono['Time'] > w_max, 'Time'] = w_max
        w_comb.loc[w_comb['Time'] > w_max, 'Time'] = w_max
        return (w_mono, w_comb)

    def __prepare_outdir(self, name):
        """Prepare output directory.

        Parameters
        ----------
        name : str
            Name of output directory.

        Returns
        -------
        str
            Output directory path.

        """
        outdir = self.outdir + name
        new_directory = Path(outdir)
        new_directory.mkdir(parents=True, exist_ok=True)
        return outdir

    def __populate_N_patients(self, ori_df):
        # TODO mark patients below the curve
        """Scale to make N patients from survival 0 to 100.

        Parameters
        ----------
        ori_df : pandas.DataFrame
            Original survival data.

        Returns
        -------
        pandas.DataFrame
            Survival data with N data points.

        """
        df = ori_df.copy()
        # add starting point (survival=100, time=0)
        df = df.append(pd.DataFrame(
            {'Survival': 100, 'Time': 0}, index=[df.shape[0]]))
        min_survival = df['Survival'].min()
        step = 100 / self.N

        f = interpolate(df, x='Survival', y='Time')  # survival -> time
        if min_survival <= 0:
            points = np.linspace(0, 100 - step, self.N)
            new_df = pd.DataFrame({'Time': f(points), 'Survival': points})
        else:
            existing_n = int(np.round((100 - min_survival) / 100 * self.N, 0))
            existing_points = np.linspace(
                100 * (self.N - existing_n) / self.N + step, 100 - step, existing_n)

            # pad patients from [0, min_survival]
            new_df = pd.DataFrame({'Time': df['Time'].max(),
                                   'Survival': np.linspace(0, 100 * (self.N - existing_n) / self.N, self.N - existing_n)})
            new_df = new_df.append(pd.DataFrame({'Survival': existing_points,
                                                 'Time': f(existing_points)}))
            new_df = new_df.round(
                {'Time': 5, 'Survival': int(np.ceil(-np.log10(step)))})
        assert new_df.shape[0] == self.N
        return new_df[['Time', 'Survival']].sort_values('Survival').reset_index(drop=True)

    def __pcp_cutoff_time(self, pcp=4):
        assert self.atrisk is not None, "No atrisk table"
        fmono = interpolate(self.mono, x='Time', y='Survival')
        fcomb = interpolate(self.comb, x='Time', y='Survival')
        pcp_mono = fmono(self.atrisk.index) / self.atrisk['control']
        pcp_comb = fcomb(self.atrisk.index) / self.atrisk['treatment']
        cutoff = min(pcp_mono[pcp_mono <= pcp].idxmax(),
                     pcp_comb[pcp_comb <= pcp].idxmax())
        return cutoff

    def __concat_weibull_tail(self, ori, weibull):
        raw = ori[ori['Time'] < self.time_max]
        if raw.iat[0, 0] - weibull.iat[raw.index.min() - 1, 0] > 0:
            surv_at_tmax = weibull[weibull['Time'] >= self.time_max].index.max()
            if np.isnan(surv_at_tmax):
                #FIXME should I output warning?
                return ori
            fill_range = np.array(range(surv_at_tmax, raw.index.min()))
            tmp = pd.DataFrame({'Survival': 100 * fill_range / self.N,
                                'Time': self.time_max},
                               index=fill_range)
            merged = pd.concat([weibull.loc[:surv_at_tmax - 1, :], tmp, raw], axis=0)
        else:
            extrap = weibull.loc[:raw.index.min() - 1, :]  # ends are inclusive
            merged = pd.concat([extrap, raw], axis=0)
        assert merged.shape[0] == self.N, "Something went wrong with concat weibull tail"
        return merged

    def change_to_save_mode(self):
        self.save_mode = True
        if self.out_name is None:
            self.outdir = self.__prepare_outdir(self.comb_name)
        else:
            self.outdir = self.__prepare_outdir(self.out_name)

    def prepare_dataframe(self, max_time):
        """Set max follow-up time for the analysis. Update mono, comb, and max_curve
        dataframes to have 'max_time' as maximum time.

        Parameters
        ----------
        max_time : float
            Max follow-up time for the analysis.

        """
        ori_max_time = min(
            self.ori_mono['Time'].max(), self.ori_comb['Time'].max())
        if max_time > ori_max_time:
            print("max_time cannot past data follow-up period. Setting according to data")
            max_time = ori_max_time
        self.mono = self.__populate_N_patients(self.ori_mono)
        self.comb = self.__populate_N_patients(self.ori_comb)

        # induce monotonicity
        self.mono.loc[:, 'Time'] = np.minimum.accumulate(self.mono['Time'].values)
        self.comb.loc[:, 'Time'] = np.minimum.accumulate(self.comb['Time'].values)
        # update time_max
        if self.atrisk is None:
            self.time_max = np.round(max_time, 5)
        else:
            self.time_max = self.__pcp_cutoff_time()

        # align max time
        self.mono.loc[self.mono['Time'] >
                      self.time_max, 'Time'] = self.time_max
        self.comb.loc[self.comb['Time'] >
                      self.time_max, 'Time'] = self.time_max
        # calculate normalized difference between two curves
        self.norm_diff = 100 * (self.comb['Time'] - self.mono['Time']).sum() / self.N
        # get weibull fitted curves
        self.w_mono, self.w_comb = self.__get_weibull_fit()

        self.mono = self.__concat_weibull_tail(self.mono, self.w_mono)
        self.comb = self.__concat_weibull_tail(self.comb, self.w_comb)

        step = 100 / self.N
        self.max_curve = pd.DataFrame({'Time': np.maximum(self.mono['Time'], self.comb['Time']),
                                       'Survival': np.linspace(0, 100 - step, self.N)})
        self.max_curve = self.max_curve.round(
            {'Time': 5, 'Survival': int(np.ceil(-np.log10(step)))})

    def fit_curve(self, prob_kind='power', prob_coef=0, prob_offset=0):
        """Fits monotherapy arm to combination arm.

        Parameters
        ----------
        prob_kind : str ('linear', 'power', 'exponential')
            Probability funciton to use when choosing patients to benefit
            (the default is 'linear').
        prob_coef : float
            Coefficent in probability function (the default is 1).
        prob_offset : float
            Offset in probablity function (the default is 0).
        """
        assert self.norm_diff >= 0, "Error: Cannot run algorithm if monotherapy is better than combination."
        rng = np.random.default_rng()

        # initialize self.record
        self.record = self.mono.copy()
        self.record.index.name = 'patient_id'
        # did patient receive benefit from added drug?
        self.record['benefit'] = 0
        # only lower-bound of the benefit is determined
        self.record['left_bound'] = 0
        # monotherapy time >= max_time (i.e. invalid result, we don't know zone)
        self.record['valid'] = 1
        self.record.loc[self.record['Time'] >= self.time_max, 'valid'] = 0
        self.record['new_t'] = self.record['Time'].copy()
        self.record['new_surv'] = self.record['Survival'].copy()

        self.info_str = 'N_{N}_prob_{prob_kind}_{prob_coef:.2f}_{prob_offset}'.format(N=self.N,
                                                                                      prob_kind=prob_kind,
                                                                                      prob_coef=prob_coef,
                                                                                      prob_offset=prob_offset)

        for i in range(self.N - 1, -1, -1):
            t = self.max_curve.at[i, 'Time']
            surv = self.max_curve.at[i, 'Survival']
            patient_pool = self.record[self.record['benefit']
                                       == 0].index.values
            pool_size = patient_pool.size

            if pool_size == 0:
                continue
            elif pool_size == 1:
                chosen_patient = patient_pool[0]
            else:
                prob = get_prob(pool_size, prob_coef,
                                prob_offset, kind=prob_kind)
                chosen_patient = rng.choice(
                    patient_pool, 1, p=prob, replace=False)[0]

            t_chosen = self.mono.at[chosen_patient, 'Time']

            if t_chosen > t:
                try:
                    chosen_patient = self.record[(self.record['benefit'] == 0) & (
                        self.record['Time'] <= t)].index.values[0]
                except IndexError:
                    continue

            self.record.at[chosen_patient, 'benefit'] = 1
            self.record.at[chosen_patient, 'new_t'] = t
            self.record.at[chosen_patient, 'new_surv'] = surv

        # add delta_t info
        self.record['delta_t'] = self.record['new_t'] - self.record['Time']
        self.record['delta_t'] = self.record['delta_t'].fillna(0)
        # left-bound delta t
        self.record.loc[self.record['new_t'] >= self.time_max, 'left_bound'] = 1
        self.record.loc[:, 'lb_delta_t'] = self.record['delta_t']
        self.record.loc[self.record['left_bound'] == 1, 'lb_delta_t'] = self.time_max - self.record[self.record['left_bound'] == 1]['Time']

        self.record_stats(prob_kind, prob_coef, prob_offset)

    def record_stats(self, prob_kind, prob_coef, prob_offset):
        """Short summary.

        """
        valid_subset = self.record[self.record['valid'] == 1]
        # benefitting percentage
        stepsize = 100 / self.N
        self.percent_certain = stepsize * \
            sum((self.record['delta_t'] > 1) & (
                self.record['left_bound'] == 0) & (self.record['valid'] == 1))
        self.percent_uncertain = stepsize * \
            sum((self.record['lb_delta_t'] > 1) & (
                self.record['left_bound'] == 1) & (self.record['valid'] == 1))
        self.percent_1mo = stepsize * sum(valid_subset['delta_t'] > 1)

        # median benefit low bound
        self.med_benefit_low = valid_subset[valid_subset['lb_delta_t'] > 1]['lb_delta_t'].median()

        # median benefit high bound
        self.med_benefit_high = valid_subset[valid_subset['delta_t'] > 1]['delta_t'].median()

        # correlation
        if self.corr == 'kendalltau':
            r2, p2 = kendalltau(valid_subset['Time'], valid_subset['delta_t'])
        elif self.corr == 'spearmanr':
            r2, p2 = spearmanr(valid_subset['Time'], valid_subset['delta_t'])
        self.corr_rho = r2
        self.corr_p = p2

    def plot_fit_curve_sanity_check(self, save=True, postfix=""):
        """Sanity check plot of monotherapy, updated, and combination arms.
        """
        try:
            mono_min_surv_idx = self.mono[self.mono['Time'] >= self.time_max].index[-1]
        except IndexError:
            mono_min_surv_idx = 0
        try:
            comb_min_surv_idx = self.comb[self.comb['Time'] >= self.time_max].index[-1]
        except IndexError:
            comb_min_surv_idx = 0
        left_bound_range = np.arange(
            mono_min_surv_idx, comb_min_surv_idx, 1) / self.N * 100

        sns.set_style('ticks')
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(self.mono.loc[mono_min_surv_idx:, 'Time'], self.mono.loc[mono_min_surv_idx:, 'Survival'],
                label='mono', color='green')
        ax.plot(self.comb.loc[comb_min_surv_idx:, 'Time'], self.comb.loc[comb_min_surv_idx:, 'Survival'],
                label="comb", color='purple')
        tmp = self.record.sort_values('new_surv')
        ax.plot(tmp['new_t'], tmp['new_surv'],
                color='orange', label="reconstructed")
        # defined region
        ax.fill_betweenx(np.arange(comb_min_surv_idx, self.N, 1) / self.N * 100,
                         self.mono.loc[comb_min_surv_idx:, 'Time'],
                         self.max_curve.loc[comb_min_surv_idx:, 'Time'],
                         color='orange', alpha=0.3)
        # left-bound region
        ax.fill_betweenx(left_bound_range,
                         self.mono.loc[mono_min_surv_idx:comb_min_surv_idx - 1, 'Time'],
                         np.repeat(self.time_max + 1, left_bound_range.size),
                         color='royalblue', alpha=0.3)
        # unknown region
        ax.axhspan(0, 100 * mono_min_surv_idx / self.N,
                   color='gray', alpha=0.3)  # gray shading for unknown
        ax.set_xlim(0, self.time_max + 1)
        ax.set_ylim(-1, 105)
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Probability (%)')
        ax.legend()
        sns.despine()
        fig.tight_layout()
        if save and self.save_mode:
            fig.savefig(f'{self.outdir}/{self.info_str}{postfix}.sanity_check.{self.fig_format}')
            plt.close()
            return

        return ax

    def plot_t_delta_t_corr(self, save=True, postfix=""):
        """Plot survival time for A against added benefit from B.

        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        valid_subset = self.record[self.record['valid'] == 1]
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
        fig.tight_layout()
        if save and self.save_mode:
            fig.savefig(f'{self.outdir}/{self.info_str}{postfix}.corr.{self.fig_format}')
            plt.close()
            return

        return ax

    def plot_benefit_distribution(self, save=True, postfix="", kind='absolute'):
        """Plot benefit distribution.

        """
        assert self.save_mode, "Cannot Save Plot: Program not in save mode."
        ### set up data
        sns.set_style('ticks')
        stepsize = 100 / self.N
        valid_subset = self.record[self.record['valid'] == 1]
        if kind == 'absolute':
            delta1 = valid_subset['lb_delta_t']  # left-bound
            delta2 = valid_subset['delta_t']  # defined
        elif kind == 'ratio':
            delta1 = 100 * valid_subset['lb_delta_t'] / valid_subset['Time']  # left-bound
            delta2 = 100 * valid_subset['delta_t'] / valid_subset['Time']  # defined
        else:
            print("ERROR: Wrong kind parameter", file=sys.stderr)
            return
        unknown = 100 * (self.N - self.record['valid'].sum()) / self.N
        delta_df1 = pd.DataFrame({'Survival': np.linspace(unknown, 100 - stepsize, delta1.size),
                                  'Time': np.sort(delta1.values)[::-1]})
        delta_df2 = pd.DataFrame({'Survival': np.linspace(unknown, 100 - stepsize, delta2.size),
                                  'Time': np.sort(delta2.values)[::-1]})

        ### plot
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

        ### add median range box
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
            if self.med_benefit_high < self.time_max:

                ax.annotate(str(round(self.med_benefit_high, 1)),
                            xy=(self.med_benefit_high, 100), xycoords='data',
                            xytext=(self.med_benefit_high + 1, 90), textcoords='data',
                            arrowprops=dict(arrowstyle="-", color='k'))
            # add "median" text
            if self.med_benefit_low > 5:
                ax.text(0.2, 100, "Median")
            else:
                ax.text(self.time_max - 5, 100, "Median")
            ax.add_patch(med_range)

            ax.set_xlabel('Added benefit from B (months)')
            ax.set_xlim(0, self.time_max)

        elif kind == 'ratio':
            ax.set_xlabel('Fold increase in time (%)')
            ax.set_xlim(0, 500)

        ax.set_ylabel('Probability (%)')
        ax.set_ylim(-1, 105)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.05), ncol=1,
                  borderaxespad=0, frameon=True)
        sns.despine()
        fig.tight_layout()
        if save and self.save_mode:
            fig.savefig(f'{self.outdir}/{self.info_str}{postfix}.distribution_{kind}.{self.fig_format}')
            plt.close()
        return ax

    def save_record(self, postfix=""):
        """Save record dataframe to csv file.

        """
        assert self.save_mode, "Cannot Save Summary Stats: Not in Save Mode"
        self.record.to_csv(f'{self.outdir}/{self.info_str}{postfix}.table.csv')

    def save_summary_stats(self, postfix=""):
        """Save summary stats information.

        """
        assert self.save_mode, "Cannot Save Summary Stats: Not in Save Mode"
        if self.atrisk is None:
            atrisk_status = "No"
        else:
            atrisk_status = "Yes"

        textstr = '\n'.join(("Date\t{}".format(date.today()),
                             "Info\t{}".format(self.info_str),
                             "Monotherapy\t{}".format(self.mono_name),
                             "Combination\t{}".format(self.comb_name),
                             "At_risk_table\t{}".format(atrisk_status),
                             "Normalized_difference\t{:.2f}".format(self.norm_diff),
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
                                 100 * np.sum(self.record['valid'] == 1) / self.N),
                             "Max_time\t{:.2f}".format(self.time_max),
                             "{0}\t{1:.2f}".format(self.corr, self.corr_rho),
                             "{0}_pvalue\t{1:.2e}".format(self.corr, self.corr_p)))

        with open(f'{self.outdir}/{self.info_str}{postfix}.summary_stats.tsv', 'w') as f:
            f.write(textstr)
