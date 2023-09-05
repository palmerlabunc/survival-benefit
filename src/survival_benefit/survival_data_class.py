from dataclasses import dataclass, field, InitVar
from typing import Optional
import pandas as pd
import numpy as np
from survival_benefit.utils import interpolate, weibull_from_digitized

@dataclass
class SurvivalData:
    name: str
    uncleaned_data: InitVar[pd.DataFrame]
    N: int
    atrisk: pd.Series | None = None
    tmax: float | None = None
    weibull_tail: bool = False
    
    original_data: pd.DataFrame = field(init=False)
    processed_data: pd.DataFrame = field(init=False)

    def __post_init__(self, uncleaned_data):
        self.original_data = SurvivalData.cleanup_digitized_km_curve(uncleaned_data)
        if self.tmax is None:
            if self.atrisk is None:
                tmax = np.round(self.original_data['Time'].max(), 5)
            else:
                tmax = self.__pcp_cutoff_time()
            self.tmax = tmax
        self.processed_data = self.__prepare_data()

    @staticmethod
    def cleanup_digitized_km_curve(input_df: pd.DataFrame) -> pd.DataFrame:
        """ Import survival data either having two columns (time, survival).

        Args:
            input_df (pd.DataFrame): survival data with two columns.

        Returns:
            pd.DataFrame: returned data frame
        """
        df = input_df.copy()
        if not ('Time' in df.columns and 'Survival' in df.columns):
            df.columns = ['Time', 'Survival']
        # if survival is in 0-1 scale, convert to 0-100
        if df['Survival'].max() <= 1.1:
            df.loc[:, 'Survival'] = df['Survival'] * 100
        
        # normalize everything to [0, 100]
        df.loc[:, 'Survival'] = 100 * df['Survival'] / df['Survival'].max()
        df.loc[df['Survival'] < 0, 'Survival'] = 0
        df.loc[df['Time'] <= 0, 'Time'] = 0.00001

        # make sure survival is in increasing order
        if df.iat[-1, 1] < df.iat[0, 1]:
            #df = df.sort_values(['Survival'], ascending=True).drop_duplicates()
            df = df.sort_values(['Survival', 'Time'], 
                                ascending=[True, False]).drop_duplicates()
            df = df.reset_index(drop=True)

        # enforce monotinicity
        df.loc[:, 'Survival'] = np.maximum.accumulate(
            df['Survival'].values)  # monotonic increasing
        df.loc[:, 'Time'] = np.minimum.accumulate(
            df['Time'].values)  # monotonic decreasing
        
        return df
    
    def set_name(self, name: str):
        self.name = name.strip()
    
    def set_N(self, n: int):
        self.N = n
        self.processed_data = self.__prepare_data()

    def set_tmax(self, time: float):
        """Set tmax to time.
        """
        self.tmax = time
        self.processed_data = self.__prepare_data()
    
    def round_time(self, decimals=5):
        self.processed_data.loc[:, 'Time'] = self.processed_data['Time'].round(decimals)

    def set_atrisk_table(self, atrisk: pd.Series):
        """
        Assign the given at risk table to self.atrisk 
        and set tmax to the time point where the percentage per patient (PCP) is below the threshold.

        Args:
            atrisk (pd.Series): at risk table for the arm. Index is time, values are number of patients at risk.
        """        
        assert isinstance(atrisk, pd.Series)
        self.atrisk = atrisk
        self.set_tmax(self.__pcp_cutoff_time())

    def add_weibull_tail(self):
        if not self.weibull_tail:
            self.weibull_tail = True
            self.processed_data = self.__concat_weibull_tail(self.processed_data)

    def __pcp_cutoff_time(self, threshold=4):
        # PCP: percentage per patient
        assert self.atrisk is not None, "No atrisk table"
        f = interpolate(self.original_data, x='Time', y='Survival')
        pcp = f(self.atrisk.index) / self.atrisk
        cutoff = pcp[pcp <= threshold].idxmax()
        return cutoff

    def __prepare_data(self):
        populated = self.__populate_N_patients()
        # induce monotonicity
        populated.loc[:, 'Time'] = np.minimum.accumulate(populated['Time'].values)
        # cut off at given tmax (all time points longer that tmax will be converted to tmax)
        populated.loc[populated['Time'] > self.tmax, 'Time'] = self.tmax
        if self.weibull_tail:
            return self.__concat_weibull_tail(populated)
        else:
            return populated

    def __populate_N_patients(self):
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
        df = self.original_data.copy()
        # add starting point (survival=100, time=0) if not already there
        if df.iat[-1, 0] != 0.00001 and df.iat[-1, 1] != 100:
            df = pd.concat([df,
                            pd.DataFrame({'Survival': 100, 'Time': 0.00001}, 
                                         index=[df.shape[0]])], 
                           axis=0)
        min_survival = df['Survival'].min()
        # remove duplicate survival values
        df = df.drop_duplicates(subset=['Survival'], keep='first')  #FIXME testing keeping the longest time
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
            new_df = pd.concat([new_df, 
                                pd.DataFrame({'Survival': existing_points,
                                              'Time': f(existing_points)})],
                                axis=0)
            new_df = new_df.round(
                {'Time': 5, 'Survival': int(np.ceil(-np.log10(step)))})
        assert new_df.shape[0] == self.N
        return new_df[['Time', 'Survival']].sort_values('Survival').reset_index(drop=True)

    def __concat_weibull_tail(self, populated: pd.DataFrame):

        assert self.weibull_tail, "weibull_tail is False"

        weibull = weibull_from_digitized(populated, self.N, self.tmax)
        original = populated[populated['Time'] < self.tmax]
        min_survival_idx = original.index.min()
        # # if weibull fit is lower at the end of follow-up
        if original.iat[0, 0] > weibull.iat[min_survival_idx - 1, 0]:
            weibull_surv_at_tmax = weibull[weibull['Time'] >= self.tmax].index.max()
            if np.isnan(weibull_surv_at_tmax):
                return populated
            fill_range = np.array(
                range(weibull_surv_at_tmax,  min_survival_idx))
            tmp = pd.DataFrame({'Survival': 100 * fill_range / self.N,
                                'Time': self.tmax},
                                index=fill_range)
            merged = pd.concat([weibull.loc[:weibull_surv_at_tmax - 1, :], tmp, original], axis=0)
        else: # if weibull fit is higher at the end of follow-up
            weibull_tail = weibull.loc[:min_survival_idx - 1, :]
            merged = pd.concat([weibull_tail, original], axis=0)

        # sanity checks
        assert merged.index.is_monotonic_increasing
        assert merged.index.is_unique
        assert merged.shape[0] == self.N
        
        return merged