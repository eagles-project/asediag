## Aerosol SE DIAGnostics (asediag)

**Aerosol process analysis in model-native Spectral Element (SE) grid**

* **Contact:** [taufiq.hassan@pnnl.gov](taufiq.hassan@pnnl.gov) 
* [Contributors](./AUTHORS.rst)

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

[Demo output page](https://compy-dtn.pnl.gov/hass877/share/aer_diag/F20TR_2016Emis_ne30pg2_def_minus_F20TR_1850Emis_ne30pg2_def/aerosol.html)

Acknowledgement
---------------
`asediag` is based on earlier developments in `NCL` and individual contributions from:
* Po-Lun Ma: [Original budget analysis and diagnostics](https://github.com/eagles-project/aerosol_diag)
* Jian Sun & Kai Zhang: [Aerosol number and species per mode budget and spatial distribution analysis](https://github.com/eagles-project/aerosol_diag/tree/jian/develop)
* Jianfeng Li: [Aerosol optics diagnostics](https://github.com/eagles-project/aerosol_diag/tree/jianfeng_update)
* Kai Zhang: [Aerosol forcing diagnostics](https://github.com/kaizhangpnl/e3sm_erf_aerosol)