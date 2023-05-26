from datetime import date
import pandas as pd
import numpy as np
import pytest
from survival_benefit.survival_data_class import SurvivalData
from survival_benefit.survival_benefit_class import SurvivalBenefit
from survival_benefit.utils import generate_weibull


class TestToySurvivalBenefit():
    @pytest.fixture(scope='class')
    def survival_benefit(self):
        mono_data = SurvivalData("mono_data", generate_weibull(100, 20, 1, 0.5, const=20), 1000, atrisk=None, tmax=15)
        comb_data = SurvivalData("comb_data", generate_weibull(100, 20, 1, 0.5, const=30), 1500, atrisk=None, tmax=18)
        return SurvivalBenefit(1000, mono_data, comb_data, atrisk=None, corr='spearmanr', 
                               outdir='./example/test_output', out_name=None, save_mode=True)
    
    def test_init(self, survival_benefit):
        assert survival_benefit.N == 1000
        assert survival_benefit.outdir == './example/test_output/comb_data'
        assert survival_benefit.atrisk is None
        assert isinstance(survival_benefit.mono_survival_data, SurvivalData)
        assert isinstance(survival_benefit.comb_survival_data, SurvivalData)
        # test tmax is aligned
        assert survival_benefit.mono_survival_data.tmax == 15
        assert survival_benefit.comb_survival_data.tmax == 15
        # test N is matched
        assert survival_benefit.mono_survival_data.N == survival_benefit.N
        assert survival_benefit.comb_survival_data.N == survival_benefit.N
        # test norm_diff is in range
        assert survival_benefit.norm_diff > 0
        # test max_curve is in correct shape
        assert survival_benefit.max_curve.shape == (survival_benefit.N, 2)
        assert (survival_benefit.max_curve['Time'] >= survival_benefit.mono_survival_data.processed_data['Time']).all()
        assert (survival_benefit.max_curve['Time'] >= survival_benefit.comb_survival_data.processed_data['Time']).all()

    def test_set_tmax(self, survival_benefit):
        survival_benefit.set_tmax(10)
        assert survival_benefit.tmax == 10
        assert survival_benefit.mono_survival_data.tmax == 10
        assert survival_benefit.comb_survival_data.tmax == 10

    def test_compute_benefit(self, survival_benefit):
        # is benefit_df correctly made
        survival_benefit.compute_benefit()
        n = survival_benefit.N
        assert isinstance(survival_benefit.benefit_df, pd.DataFrame)
        assert survival_benefit.benefit_df.shape == (n, 9)

        # can it determine the right number of valid data points
        valid_prop = survival_benefit.benefit_df['valid'].sum() / n
        assert valid_prop > 0.5 and valid_prop < 1.0

        # can it determine left-bound benefit correctly
        lb_prob = survival_benefit.benefit_df['left_bound'].sum() / n
        assert lb_prob > 0 and lb_prob < 1
        #FIXME need better assert
        assert survival_benefit.info_str is not None
        assert survival_benefit.med_benefit_low is not None
        assert survival_benefit.med_benefit_high is not None
        assert survival_benefit.percent_certain is not None
        assert survival_benefit.percent_uncertain is not None
        assert survival_benefit.percent_1mo > 0 and survival_benefit.percent_1mo < 100
        assert survival_benefit.corr_rho > -1 and survival_benefit.corr_rho < 1
        assert survival_benefit.corr_p > 0 and survival_benefit.corr_p < 1

        # make sure the original curve can be reconstructed
        valid_ids = survival_benefit.benefit_df[~survival_benefit.benefit_df['left_bound'] & survival_benefit.benefit_df['valid']].index
        reconstructed = survival_benefit.benefit_df.reindex(valid_ids).sort_values('new_surv')['new_t'].values
        
        ori_ids = (survival_benefit.benefit_df.reindex(valid_ids)['new_surv'].sort_values() * n / 100).astype(int).values
        original = survival_benefit.max_curve.reindex(ori_ids)['Time'].values
        assert np.array_equal(reconstructed, original)

    def test_plot_compute_benefit_sanity_check(self, survival_benefit):
        survival_benefit.compute_benefit()
        ax = survival_benefit.plot_compute_benefit_sanity_check(save=True)
        #assert ax is None

    def test_plot_t_delta_t_corr(self, survival_benefit):
        survival_benefit.compute_benefit()
        ax = survival_benefit.plot_t_delta_t_corr(save=True)
        #assert ax is None

    def test_plot_benefit_distribution(self, survival_benefit):
        survival_benefit.compute_benefit()
        ax = survival_benefit.plot_benefit_distribution(save=True)
        #assert ax is None
