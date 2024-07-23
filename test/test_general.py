import pytest
import importlib
from pathlib import Path
import configparser
import os
import warnings

@pytest.fixture(scope="module")
def base_path():
    return Path(__file__).parent.parent

def test_import_main_modules():
    """Test if main modules can be imported without errors."""
    modules = [
        'asediag',
        'src.aer_budget_analysis',
        'src.gen_budgets',
        'src.gen_forcings',
        'src.gen_spatial_distr',
        'src.gen_vertical_distr'
    ]
    for module in modules:
        importlib.import_module(module)

def test_main_script_execution(base_path):
    """Test if the main script can be executed."""
    script_path = base_path / 'asediag.py'
    result = os.system(f'python3 {script_path} -h')
    assert result == 0, "Execution of the main script failed"

def test_gen_budgets_function():
    """Test basic function execution in gen_budgets module."""
    from src.gen_budgets import GenAerosolBudgets
    try:
        GenAerosolBudgets(0,'bc','path1','path2','case1','case2','path','region','loc','eam',False,False)
    except Exception as e:
        pytest.fail(f"Execution of gen_budgets function failed: {e}")

def test_batch_config_parsing(base_path):
    """Test if the batch configuration file can be parsed."""
    config_path = base_path / 'batch_config.ini'
    assert config_path.exists(), "Configuration file does not exist"

    config = configparser.ConfigParser()
    try:
        config.read(config_path)
    except Exception as e:
        pytest.fail(f"Parsing configuration file failed: {e}")

def test_cli_script_execution(base_path):
    """Test if the CLI script can be executed."""
    cli_script_path = base_path / 'src' / 'aer_diag_cli.py'
    result = os.system(f'python3 {cli_script_path} --help')
    assert result == 0, "Execution of the CLI script failed"

def test_no_deprecation_warnings():
    """Test that the code does not produce deprecation warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import src.aer_budget_analysis  # Import the module to trigger warnings
        assert len(w) == 0, "Deprecation warnings found"