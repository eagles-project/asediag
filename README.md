## Aerosol Diagnostics on Model Native Grid

**Aerosol process analysis in model-native Spectral Element (SE) grid**

Aerosol Diagnostics on Model Native Grid is a Python-based tool designed for diagnosing aerosol processes in E3SM. It is particularly designed to analyze simulation data on the model-native spectral element (SE) grid, so it is also known as Aerosol SE Diagnostics or “asediag”. It includes functionalities for aerosol budget analysis, optical property diagnostics, and decomposed anthropogenic aerosol radiative forcing analysis. Data analysis on the native grid enhances the efficiency and scalability for high-resolution applications. On the other hand, asediag also supports analysis on the regular latitude-longitude (RLL) grid, so the modelers can flexibly choose the data to analyze. It operates in batch mode with multiprocessing capabilities, allowing rapid processing of large datasets. 

**Primary Developer & Maintainer:** Taufiq Hassan

**For Questions:** [taufiq.hassan@pnnl.gov](taufiq.hassan@pnnl.gov)

[**Contributors**](./AUTHORS.rst)

Installation
------------

Git clone the repo
```bash
git clone git@github.com:TaufiqHassan/asediag.git
```

### Submit as a batch job

`asediag` works with e3sm_unifed environment in *Compy* and *Perlmutter*.

Make adjustments in the batch configuration file `batch_config.ini`.

Submit the batch jobs with:
```bash
python submit_asediag_batches.py
```
### Example 

[Demo output page](https://compy-dtn.pnl.gov/hass877/share/aer_diag/F20TR_2016Emis_ne30pg2_def_minus_F20TR_1850Emis_ne30pg2_def_Global/aerosol.html)

Acknowledgement
---------------
`asediag` is based on earlier developments in `NCL` and individual contributions from:
* Po-Lun Ma: [Original budget analysis and diagnostics](https://github.com/eagles-project/aerosol_diag)
* Jian Sun & Kai Zhang: [Aerosol number and species per mode budget and spatial distribution analysis](https://github.com/eagles-project/aerosol_diag/tree/jian/develop)
* Jianfeng Li: [Aerosol optics diagnostics](https://github.com/eagles-project/aerosol_diag/tree/jianfeng_update)
* Kai Zhang: [Aerosol forcing diagnostics](https://github.com/kaizhangpnl/e3sm_erf_aerosol)

**Note:** Constantly undegoing small patchworks and further developments.
