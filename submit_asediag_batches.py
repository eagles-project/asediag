'''
This will produce the batch scripts based on the 
user provided info in the config file (e.g., batch_config.ini).
Submit batch scripts: python submit_asediag_batches.py
'''
import configparser
import shutil
import pkg_resources
from subprocess import Popen, PIPE, STDOUT

def exec_shell(cmd):
    '''func to execute shell commands'''
    cmd_split = cmd.split(' ')
    p = Popen(cmd_split, stdout=PIPE, stdin=PIPE, stderr=STDOUT, universal_newlines=True)
    op, _ = p.communicate()

## parsing config file to read user-defined variables
## Make sure the batch file is in the same directory
config = configparser.ConfigParser(allow_no_value=True)
config.read('batch_config.ini')

account = config.get('BATCH','account')
partition = config.get('BATCH','partition')
env = config.get('ENV','env')
asediag_dir = config.get('CMD','asediag_dir')
inDirectory1 = config.get('CMD','inDirectory1')
inDirectory2 = config.get('CMD','inDirectory2')
outDirectory = config.get('CMD','outDirectory')
model = config.get('CMD','model')
walltime = config.get('CMD','walltime')
scrip_file = config.get('CMD','scrip_file')
diags = config.get('CMD','diags')
region = config.get('CMD','region')
case1 = config.get('CMD','case1')
case2 = config.get('CMD','case2')

if case1==None:
    case1 = inDirectory1.strip().split('/')[-3]
if case2==None:
    case2 = inDirectory2.strip().split('/')[-3]
    
## Dictionary for different diagnostics and their relevant command line inputs
## For more info check the with help command: python asediag.py -h
itemDict = {'latlon':' -vlist','tables':' -tab -hplot','forcings':' -forcing -hplot',\
            'surface':' -pval 0','200':' -pval 200','500':' -pval 500','850':' -pval 850'}

for item in diags.split(','):
    ## defines this script path (i.e. asediag)
    resource_package = __name__
    resource_path = 'asediag/batch_script/get_sediag.sh'
    tmp = pkg_resources.resource_filename(resource_package, resource_path)
    ## copying the template batch file for each diagnostics (items)
    shutil.copy(tmp, outDirectory+'/get_sediag_'+item+'.sh')
    ## repacing template with actual info from config
    with open(outDirectory+'/get_sediag_'+item+'.sh','r') as file:
        filedata = file.read()
        filedata = filedata.replace('<account>',account)
        filedata = filedata.replace('<partition>',partition)
        filedata = filedata.replace('<env>',env)
        filedata = filedata.replace('<asediag_dir>',asediag_dir)
        filedata = filedata.replace('<dir1>',inDirectory1)
        filedata = filedata.replace('<dir2>',inDirectory2)
        filedata = filedata.replace('<outDir>',outDirectory)
        filedata = filedata.replace('<case1>',case1)
        filedata = filedata.replace('<case2>',case2)
        filedata = filedata.replace('<model>',model)
        filedata = filedata.replace('<wallMin>',walltime)
        filedata = filedata.replace('<region>',region)
        filedata = filedata.replace('<scrip_file>','<scrip_file>'+itemDict[item])
        filedata = filedata.replace('<scrip_file>',scrip_file)
    ## writing out to out directory
    with open(outDirectory+'/get_sediag_'+item+'.sh','w') as file:
        file.write(filedata)
    ## submitting the batch jobs
    exec_shell(f'sbatch {outDirectory}/get_sediag_'+item+'.sh')
    
