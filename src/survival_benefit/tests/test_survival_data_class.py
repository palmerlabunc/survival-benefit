import pandas as pd
import numpy as np
import pytest
from survival_benefit.survival_data_class import SurvivalData
from survival_benefit.utils import generate_weibull

####################################
# TODO                             #
####################################
# 1. test adding weibull tail

@pytest.fixture(scope='function')
def toy_survivalData():
    df = generate_weibull(1000, 20, 1, 0.5)
    return SurvivalData("toy_data", df, 2000, 
                        weibull_tail=False, atrisk=None, tmax=None)


def test_post_init(toy_survivalData):
    assert toy_survivalData.tmax == 20

    original_data = toy_survivalData.original_data
    assert isinstance(original_data, pd.DataFrame)
    assert original_data['Time'].max() == toy_survivalData.tmax
    assert original_data['Survival'].max() == 100
    assert original_data['Survival'].min() >= 0
    assert original_data['Time'].is_monotonic_decreasing
    assert original_data['Survival'].is_monotonic_increasing

    processed_data = toy_survivalData.processed_data
    assert isinstance(processed_data, pd.DataFrame)
    assert processed_data['Time'].max() == toy_survivalData.tmax
    assert processed_data.shape[0] == 2000
    assert processed_data['Time'].is_monotonic_decreasing
    assert processed_data['Survival'].is_monotonic_increasing


def test_set_tmax(toy_survivalData):
    toy_survivalData.set_tmax(15)
    assert toy_survivalData.tmax == 15
    assert toy_survivalData.processed_data['Time'].max() == 15


def test_set_N(toy_survivalData):
    toy_survivalData.set_N(10000)
    assert toy_survivalData.N == 10000
    assert toy_survivalData.processed_data.shape[0] == 10000
    assert toy_survivalData.tmax == 20