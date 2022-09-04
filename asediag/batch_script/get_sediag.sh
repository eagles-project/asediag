#!/bin/bash -l
#SBATCH --job-name=sediag
#SBATCH --output=sediag.o%j
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --time=<wallMin>
#SBATCH --partition=<partition>

source /share/apps/E3SM/conda_envs/base/etc/profile.d/conda.sh
conda activate <env>
python <asediag_dir>asediag.py -dir1 <dir1> -dir2 <dir2> -path <outDir> -cs1 <case1> -cs2 <case2> -m <model> -reg <region> -scrip <scrip_file> 
