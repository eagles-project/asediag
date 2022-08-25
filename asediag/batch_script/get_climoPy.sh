#!/bin/bash -l
#SBATCH --job-name=pyclimo
#SBATCH --output=climopy.o%j
#SBATCH --account=esmd
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=short

source /share/apps/E3SM/conda_envs/base/etc/profile.d/conda.sh
conda activate work
python /qfs/people/hass877/MODS/pyclimo/pyclimo.py -c F20TR_v2_ndg_MERRA2_Defdata -s 2015 -dir /compyfs/hass877/e3sm_scratch/F20TR_v2_ndg_MERRA2_Defdata/run/ 
