import pandas as pd
import yaml
import pytest

from src.processing.process_CTRPv2_data import create_drug_info_df, create_cell_info_df


def test_create_drug_info_df():
    drug_info = create_drug_info_df()
    assert isinstance(drug_info, pd.DataFrame)
    assert 'Database' in drug_info.columns
    assert drug_info['Database'].unique() == ['CTRPv2']
    assert len(drug_info) > 0


def test_create_cell_info_df():
    cell_info = create_cell_info_df()
    assert isinstance(cell_info, pd.DataFrame)
    assert 'Dataset' in cell_info.columns
    assert cell_info['Dataset'].unique() == ['CTRPv2']
    assert 'Drug Data Available in Dataset' in cell_info.columns
    assert (cell_info['Drug Data Available in Dataset'] == 'y').all()
    assert 'Cancer_Type_HH' in cell_info.columns
    assert len(cell_info) > 0
