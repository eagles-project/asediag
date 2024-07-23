import pytest
import importlib
from pathlib import Path
import xarray as xr
import numpy as np
import configparser
import os
import warnings

from src.aer_budget_analysis import AerosolBudgetCalculator

@pytest.fixture(scope="module")
def base_path():
    return Path(__file__).parent.parent

def test_aer_budget_factors():
    """Test factors."""
    factors = {'fact': 1e-09,
                'grav': 9.806,
                'factaa': 1.909664563268017e-21,
                'factaaa': 1.9926934573231485e-22,
                'factbb': 0.031536,
                'factcc': 0.008793333704347827,
                'factdd': 2.788347826086957e-10,
                'fact_kgpkg_kgpcm3': 1.2922610580794262e-06,
                'fact_kgpcm3_ugpm3': 1292261058.079426}
    abc = AerosolBudgetCalculator('','','','')
    assert abc.calculate_factors() == factors

def test_aer_budget_region_selection():
    """Test region selection."""
    path = ''
    case = ''
    ts = ''
    aer = ''
    abc = AerosolBudgetCalculator(path,case,ts,aer,reg='NA')
    abc.set_region()
    assert (abc.lat1,abc.lat2,abc.lon1,abc.lon2) == (15.0, 72.0, -167.0, -50.0)

def test_aer_budget_regional_mean(base_path):
    """Test region selection."""
    path = base_path / 'data'
    case = ''
    ts = ''
    aer = 'bc'
    filename = path / 'TestSEdata.nc'
    data = xr.open_dataset(filename)
    lon = data['lon'].values
    lon[lon > 180.] -= 360.
    lat = data['lat']
    area = data['area']
    abc = AerosolBudgetCalculator(path,case,ts,aer,data=data,lon=lon,lat=lat,area=area,reg='CONUS')
    abc.set_region()
    vdata = data['bc_a1']
    mean = abc.calculate_mean(vdata,'bc')
    assert np.round(mean.values*1e10,2) == 0.15

def test_aer_budget_df(base_path):
    path = Path('./data')
    case = ''
    ts = ''
    aer = 'bc'
    filename = path / 'TestSEdata.nc'
    data = xr.open_dataset(filename)
    lon = data['lon'].values
    lon[lon > 180.] -= 360.
    lat = data['lat']
    area = data['area']
    data['SFbc_a1'] = data['bc_a1']
    data = data[['area', 'PS', 'hyai', 'hybi', 'P0', 'LANDFRAC', 'lon', 'lat', 'lev', 'ilev', 'year', 'SFbc_a1']]
    abc = AerosolBudgetCalculator(path,case,ts,aer,data=data,lon=lon,lat=lat,area=area,reg='CONUS')
    abc.factors = abc.calculate_factors()
    abc.get_data_params()
    abc.set_region()
    avar, cvar = abc.get_var_lists()
    df, nvdata = abc.process_data(avar,cvar)
    df['season'] = np.nan
    df = abc.finalize_dataframe(df)
    np.round(df['bc'].loc['surface emission'],2) == 19.14