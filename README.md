## Aerosol SE DIAGnostics (asediag)

**Aerosol process analysis in model-native Spectral Element (SE) grid**

* Based on original NCL package from Po-Lun.
* Radiative Forcing analysis from Kai Zhang.

Installation
------------

Git clone the repo
```bash
git clone git@github.com:TaufiqHassan/asediag.git
```

### Submit as a batch job

`asediag` works with e3sm_unifed environment

Make adjustments in the batch configuration file `batch_config.ini`

```bash
## Config file for running all batch scripts on different nodes
[BATCH]
account = esmd
partition = short

[ENV]
## Default e3sm unified env for compy
source = /share/apps/E3SM/conda_envs/load_latest_e3sm_unified_compy.sh
## source for Cori: /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_cori-haswell.sh
## if using a user specific environment, source conda and activate user environment (env = work)
## Otherwise, keep as is.
env =
## compy source: /share/apps/E3SM/conda_envs/base/etc/profile.d/conda.sh
## Cori source: /global/common/software/e3sm/anaconda_envs/base/etc/profile.d/conda.sh

[CMD]
## asediag directory
asediag_dir = /qfs/people/hass877/MODS/asediag/
## Control case directory
inDirectory1 = /compyfs/hass877/e3sm_scratch/clim_out/F20TR_v2_ndg_PDemis_Defdata/
## Test case directory
inDirectory2 = /compyfs/hass877/e3sm_scratch/clim_out/F20TR_v2_ndg_PDemis_SEdata/
## asediag output directory (will also contain the batch scripts and the sediag logs)
outDirectory = /compyfs/www/hass877/share/aer_diag
## No values indicate inDirectory as outDirectory
## or put full directory path (ex: outDirectory = some/output/dir)
case1 = F20TR_v2_ndg_PDemis_Defdata
case2 = F20TR_v2_ndg_PDemis_SEdata
## No values indicates climo files in case run directory
## and case names are inferred from the input directories
## model name cam for E3SMv1
model = eam
## Curretly supported regions Global, NA, CONUS, EUS, NH, SH, NH_pole, SH_pole, Tropics
region = Global
walltime = 00:15:00
## For default ne30, 10 min is enough. For RRM, it can be 10-30 min depending
## on the freq (freq = mon takes longer)
## SCRIP file required for spatial maps in SE grid
## For lat-lon keep as is
scrip_file = /compyfs/www/hass877/share/emis_data/DECK120_to_SE/ne30pg2_SCRIP.nc
## Available diags options: latlon,tables,extra,forcings,surface,200,500,850,zonal
diags = latlon,tables,extra,forcings,surface,200,500,850,zonal
## Run this config by: python submit_asediag_batches.py
```