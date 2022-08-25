#!/bin/bash -l
#SBATCH --job-name=sediag
#SBATCH --output=sediag.o%j
#SBATCH --account=esmd
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --partition=short

source /share/apps/E3SM/conda_envs/base/etc/profile.d/conda.sh
conda activate work
python /qfs/people/hass877/MODS/aer_diag_SE/asediag.py -dir1 /compyfs/hass877/e3sm_scratch/F20TR_v2_ndg_ERA5_SEdata_NA_RRM_PI/run/ -dir2 /compyfs/hass877/e3sm_scratch/F20TR_v2_ndg_ERA5_SEdata_NA_RRM/run/  -forcing -hplot
