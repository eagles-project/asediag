#!/bin/bash -l
#SBATCH --job-name=asediag
#SBATCH --output=<logDir>/sediag.<option>.o%j
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --time=<wallMin>
#SBATCH --qos=<partition>
#SBATCH --constraint=cpu

source <source>
# user-defined environment
python <asediag_dir>asediag.py -dir1 <dir1> -dir2 <dir2> -path <outDir> -cs1 <case1> -cs2 <case2> -m <model> -reg <region> -scrip <scrip_file> 
