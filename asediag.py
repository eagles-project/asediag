from src.aer_diag_cli import main

from src.aer_budget_analysis import AerosolBudgetCalculator
from src.gen_budgets import GenAerosolBudgets
from src.gen_spatial_distr import GenSpatialData, SpatialMapGenerator
from src.gen_vertical_distr import GenVerticalData, GetVerticalProfiles
from src.gen_forcings import ForcingAnalyzer

from src.utils.aerdiag_plots import get_plots
from src.utils.nclCols import amwg256_map, BlueWhiteOrangeRed_map

if __name__=='__main__':
    main()
