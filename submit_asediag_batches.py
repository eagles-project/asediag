import configparser
import shutil
import pkg_resources
from subprocess import Popen, PIPE, STDOUT

def exec_shell(cmd):
    cmd_split = cmd.split(' ')
    p = Popen(cmd_split, stdout=PIPE, stdin=PIPE, stderr=STDOUT, universal_newlines=True)
    op, _ = p.communicate()

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

case1 = inDirectory1.strip().split('/')[-3]
case2 = inDirectory2.strip().split('/')[-3]

itemDict = {'latlon':' -vlist','tables':' -tab -hplot','forcings':' -forcing -hplot',\
            'surface':' -pval 0','200':' -pval 200','500':' -pval 500','850':' -pval 850'}

for item in diags.split(','):
    resource_package = __name__
    resource_path = 'asediag/batch_script/get_sediag.sh'
    tmp = pkg_resources.resource_filename(resource_package, resource_path)
    shutil.copy(tmp, outDirectory+'/get_sediag_'+item+'.sh')
    with open(outDirectory+'/get_sediag_'+item+'.sh','r') as file:
        filedata = file.read()
        filedata = filedata.replace('<account>',account)
        filedata = filedata.replace('<partition>',partition)
        filedata = filedata.replace('<env>',env)
        filedata = filedata.replace('<asediag_dir>',asediag_dir)
        filedata = filedata.replace('<dir1>',inDirectory1)
        filedata = filedata.replace('<dir2>',inDirectory2)
        filedata = filedata.replace('<outDir>',outDirectory)
        filedata = filedata.replace('<model>',model)
        filedata = filedata.replace('<wallMin>',walltime)
        filedata = filedata.replace('<scrip_file>','<scrip_file>'+itemDict[item])
        filedata = filedata.replace('<scrip_file>',scrip_file)

    with open(outDirectory+'/get_sediag_'+item+'.sh','w') as file:
        file.write(filedata)
    exec_shell(f'sbatch {outDirectory}/get_sediag_'+item+'.sh')
    
