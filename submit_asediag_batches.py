'''
This will produce the batch scripts based on the 
user provided info in the config file (e.g., batch_config.ini).
Submit batch scripts: python submit_asediag_batches.py
'''
#import configparser
from six.moves import configparser
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
source = config.get('ENV','source')
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
vlist = config.get('EXTRA','vlist')
vlist = vlist.replace('\n','').replace(' ','')
vunit = config.get('EXTRA','vunit')
vunit = vunit.replace('\n','').replace(' ','')
plist = config.get('EXTRA','plist')
plist = plist.replace('\n','').replace(' ','')
punit = config.get('EXTRA','punit')
punit = punit.replace('\n','').replace(' ','')
loc = config.get('EXTRA','ProfLoc')

if case1==None:
    case1 = inDirectory1.strip().split('/')[-3]
if case2==None:
    case2 = inDirectory2.strip().split('/')[-3]

## Actual output path directory
path = str(outDirectory)+'/'+case2+'_minus_'+case1
print('\nSelected output directoy:',path)
print('\nAll shell scripts and log files will be stored here.')
## copying the template content to out dir (i.e. outpath)
## Getting rid of template directory would be best
resource_package = __name__
resource_path = 'asediag/template'
tmp = pkg_resources.resource_filename(resource_package, resource_path)
try:
    shutil.copytree(tmp, path)
except:
    print('\nCan not create directory:',path)
    pass
with open(tmp+'/aerosol.html','r') as file:
    filedata = file.read()
    filedata = filedata.replace('F20TR_v2_ndg_ERA5_SEdata_NA_RRM',case2)
    filedata = filedata.replace('F20TR_v2_ndg_ERA5_Defdata_NA_RRM',case1)
with open(path+'/aerosol.html','w') as file:
    file.write(filedata)
    
## Dictionary for different diagnostics and their relevant command line inputs
## For more info check the with help command: python asediag.py -h
itemDict = {'extra':' -vlist '+vlist+' -vunit '+vunit+' -hplot','tables':' -tab -hplot',\
            'forcings':' -forcing -hplot','surface':' -pval 0','200':' -pval 200',\
            '500':' -pval 500','850':' -pval 850','zonal':' -prof -hplot',\
            'extraprofs':' -eprof '+plist+' -gunit '+punit+' -hplot','latlon':''}

for item in diags.split(','):
    ## defines this script path (i.e. asediag)
    resource_package = __name__
    resource_path = 'asediag/batch_script/get_sediag.sh'
    tmp = pkg_resources.resource_filename(resource_package, resource_path)
    ## copying the template batch file for each diagnostics (items)
    shutil.copy(tmp, path+'/get_sediag_'+item+'.sh')
    ## repacing template with actual info from config
    with open(path+'/get_sediag_'+item+'.sh','r') as file:
        filedata = file.read()
        filedata = filedata.replace('<account>',account)
        filedata = filedata.replace('<partition>',partition)
        filedata = filedata.replace('<source>',source)
        if env != '':
            filedata = filedata.replace('# user-defined environment','conda activate '+env)
        filedata = filedata.replace('<asediag_dir>',asediag_dir)
        filedata = filedata.replace('<dir1>',inDirectory1)
        filedata = filedata.replace('<dir2>',inDirectory2)
        filedata = filedata.replace('<outDir>',outDirectory)
        filedata = filedata.replace('<logDir>',path)
        filedata = filedata.replace('<case1>',case1)
        filedata = filedata.replace('<case2>',case2)
        filedata = filedata.replace('<model>',model)
        filedata = filedata.replace('<wallMin>',walltime)
        filedata = filedata.replace('<region>',region)
        filedata = filedata.replace('<scrip_file>','<scrip_file>'+itemDict[item])
        if ((item=='zonal') or (item=='extraprofs')) and (((loc!=None) and loc!='')):
            filedata = filedata.replace('<scrip_file>'+itemDict[item],'<scrip_file>'+itemDict[item]+' -loc '+loc)          
        filedata = filedata.replace('<scrip_file>',scrip_file)
    ## writing out to out directory
    with open(path+'/get_sediag_'+item+'.sh','w') as file:
        file.write(filedata)
    ## submitting the batch jobs
    job_string = 'sbatch '+path+'/get_sediag_'+item+'.sh'
    exec_shell(job_string)
    
