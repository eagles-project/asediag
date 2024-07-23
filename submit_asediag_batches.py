"""
This script generates batch scripts based on user-provided information in the config file (e.g., batch_config.ini).
To submit batch scripts, run: python submit_asediag_batches.py
"""

import shutil
import logging
import importlib.resources as pkg_resources
from six.moves import configparser

from src.utils.asediag_utils import exec_shell, setup_output_directory

# Configure logging to log messages to a file
log_file = 'log.asediag'
logging.basicConfig(level=logging.INFO, handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])

# Parsing config file to read user-defined variables
config = configparser.ConfigParser(allow_no_value=True)
config.read('batch_config.ini')

def get_config_value(section, key, default=None):
    return config.get(section, key, fallback=default).replace('\n', '').replace(' ', '')

# Reading configuration values
account = config.get('BATCH', 'account')
partition = config.get('BATCH', 'partition')
env = config.get('ENV', 'env')
source = config.get('ENV', 'source')
asediag_dir = config.get('CMD', 'asediag_dir')
inDirectory1 = config.get('CMD', 'inDirectory1')
inDirectory2 = config.get('CMD', 'inDirectory2')
outDirectory = config.get('CMD', 'outDirectory')
model = config.get('CMD', 'model')
walltime = config.get('CMD', 'walltime')
scrip_file = config.get('CMD', 'scrip_file')
diags = config.get('CMD', 'diags')
region = config.get('CMD', 'region')
case1 = config.get('CMD', 'case1')
case2 = config.get('CMD', 'case2')
vlist = get_config_value('EXTRA', 'vlist')
vunit = get_config_value('EXTRA', 'vunit')
plist = get_config_value('EXTRA', 'plist')
punit = get_config_value('EXTRA', 'punit')
loc = config.get('EXTRA', 'ProfLoc')

case1 = case1 or inDirectory1.strip().split('/')[-3]
case2 = case2 or inDirectory2.strip().split('/')[-3]

# Setting output path and copying template frontend html
path = setup_output_directory(outDirectory, case1, case2, region)
package_name = 'src.utils'
with pkg_resources.open_text(package_name, 'aerosol_temp.html') as file:
    filedata = file.read()
    filedata = filedata.replace('F20TR_v2_ndg_ERA5_SEdata_NA_RRM', f'{case2}_{region}')
    filedata = filedata.replace('F20TR_v2_ndg_ERA5_Defdata_NA_RRM', f'{case1}_{region}')
with open(path / 'aerosol.html', 'w') as file:
    file.write(filedata)

# Dictionary for different diagnostics and their relevant command line inputs
itemDict = {
    'extra': f' -vlist {vlist} -vunit {vunit} -hplot',
    'tables': ' -tab -hplot',
    'forcings': ' -forcing -hplot',
    'surface': ' -pval 1000',
    '200': ' -pval 200',
    '500': ' -pval 500',
    '850': ' -pval 850',
    'zonal': ' -prof -hplot',
    'extraprofs': f' -eprof {plist} -gunit {punit} -hplot',
    'latlon': '',
    'bplots': ' -sp -hplot'
}

for item in diags.split(','):
    item = item.strip()
    tmp = 'src.batch_script'
    # Copying the template batch file for each diagnostic item
    dst = path / f'get_sediag_{item}.sh'
    with pkg_resources.path(tmp, 'get_sediag.sh') as src:
        shutil.copy(src, dst)
    
    # Replacing template placeholders with actual info from config
    with open(path / f'get_sediag_{item}.sh', 'r') as file:
        filedata = file.read()
        filedata = filedata.replace('<account>', account)
        filedata = filedata.replace('<partition>', partition)
        filedata = filedata.replace('<source>', source)
        if env:
            filedata = filedata.replace('# user-defined environment', f'conda activate {env}')
        filedata = filedata.replace('<asediag_dir>', asediag_dir)
        filedata = filedata.replace('<dir1>', inDirectory1)
        filedata = filedata.replace('<dir2>', inDirectory2)
        filedata = filedata.replace('<outDir>', outDirectory)
        filedata = filedata.replace('<logDir>', str(path))
        filedata = filedata.replace('<case1>', case1)
        filedata = filedata.replace('<case2>', case2)
        filedata = filedata.replace('<model>', model)
        filedata = filedata.replace('<wallMin>', walltime)
        filedata = filedata.replace('<option>', item)
        filedata = filedata.replace('<region>', region)
        filedata = filedata.replace('<scrip_file>','<scrip_file>'+itemDict[item])
        if ((item=='zonal') or (item=='extraprofs')) and (((loc!=None) and loc!='')):
            filedata = filedata.replace('<scrip_file>'+itemDict[item],'<scrip_file>'+itemDict[item]+' -loc '+loc) 
        filedata = filedata.replace('<scrip_file>',scrip_file)

    # Writing the modified batch script to the output directory
    with open(path / f'get_sediag_{item}.sh', 'w') as file:
        file.write(filedata)
    
    # Submitting the batch jobs
    job_string = f'sbatch {path}/get_sediag_{item}.sh'
    exec_shell(job_string)

# Move the log file to the output directory
shutil.move(log_file, path / log_file)
