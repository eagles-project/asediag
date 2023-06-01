## Aerosol SE DIAGnostics (asediag)

**Aerosol process analysis in model-native Spectral Element (SE) grid**

* Based on `original NCL package from Po-Lun`_.
* Radiative Forcing analysis from Kai Zhang.

Installation
------------

Works with e3sm_unifed environment

On compy: 
```bash
source /share/apps/E3SM/conda_envs/load_latest_e3sm_unified_compy.sh
```

On Cori: 
```bash
source /global/common/software/e3sm/anaconda_envs/load_latest_e3sm_unified_cori-haswell.sh
```

For others, use the YAML file provided to create a virtual conda enviroment (genv)

`conda env create -f environment.yml`

And then activate genv to use ggen

`conda activate genv`