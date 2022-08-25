import time
import argparse
from itertools import product
import multiprocessing as mp
from pathlib import Path
import pkg_resources
import shutil

from asediag.aerosol_diag_SEgrid import gather_data, get_map, get_all_tables, get_forcing_df


def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-pval", help="p-level", default=None)
    parser.add_argument("-dir1", help="case 1 directory", default=None)
    parser.add_argument("-dir2", help="case 2 directory", default=None)
    parser.add_argument("-path", help="analysis output directory", default=Path('.').absolute())
    parser.add_argument("-cs1", help="case1", default=None)
    parser.add_argument("-cs2", help="case2", default=None)
    parser.add_argument("-m", help="model", default='eam')
    parser.add_argument("-reg", help="region", default=None)
    parser.add_argument("-loc", help="location", default=None)
    parser.add_argument("-vlist", help="plot vlist variables", action='store_true', default=None)
    parser.add_argument("-tab", help="get budget tables", action='store_true', default=None)
    parser.add_argument("-forcing", help="get forcing analysis", action='store_true', default=None)
    parser.add_argument("-hplot", help="horizontal plots", action='store_true', default=None)
    parser.add_argument("-scrip", help="scrip file", \
                        default='/compyfs/www/hass877/share/emis_data/DECK120_to_SE/northamericax4v1pg2_scrip.nc')
   
    args = parser.parse_args()
    pv = args.pval
    path1 = args.dir1
    path2 = args.dir2
    outpath = args.path
    model = args.m
    region = args.reg
    local = args.loc
    vl = args.vlist
    tb = args.tab
    hp = args.hplot
    sc = args.scrip
    rf = args.forcing
    case1 = args.cs1
    case2 = args.cs2
    
    ## Add extra variables and units below here as necessaray
    vlist = ['AODVIS','AODABS','AODALL','AODBC','AODDUST','AODPOM','AODSO4','AODSOA',\
        'AODSS','BURDENBC','BURDENSO4','BURDENDUST','BURDENPOM','BURDENSOA',\
         'BURDENMOM','BURDENSEASALT','BURDEN1','BURDEN2','BURDEN3','BURDEN4',\
        'FSNS','FSNSC','FSNT','FSNTC','FLNS','FLNSC','FLNT','FLNTC']
    vunits = ['[unitless]']*9+['[kg m$^{-2}$]']*11+['[W m$^{-2}$]']*8
    ## Add extra variables and units above here as necessaray
    if case1 == None:
        case1 = path1.strip().split('/')[-3]
    if case2 == None:
        case2 = path2.strip().split('/')[-3]
    path = str(outpath)+'/'+case2+'_minus_'+case1
    print('\nSelected output directoy:',path)
    ## copying the adiags content to out dir (i.e. outpath)
    resource_package = __name__
    resource_path = 'adiags'
    try:
        tmp = pkg_resources.resource_filename(resource_package, resource_path)
        print(tmp)
        shutil.copytree(tmp, path)
    finally:
        pkg_resources.cleanup_resources()
        
    aer_list = ['bc','so4','dst','mom','pom','ncl','soa','num']    
    start = time.perf_counter()
    # if __name__ == '__main__':
    if hp == None:
        print('\nPlotting is muted\n')
        for aer in aer_list[:]:
            print('getting data\n')
            print(path1,path2)
            print(path1.strip().split('/')[-3],pv)
            aa=gather_data(path1,aer,path1.strip().split('/')[-3],plev=pv)
            bb=gather_data(path2,aer,path2.strip().split('/')[-3],plev=pv)
            aa[0].load()
            bb[0].load()
            aa[1].load()
            bb[1].load()
            lon = aa[5].round()
            lat = aa[6].round()
            print('Loaded data\n')
            diff = bb[0]-aa[0]
            rel = (diff/abs(aa[0]))*100
            processes=[]
            for ind,var in product([0,1,2],aa[2]):
                p = mp.Process(target=get_map,args=[aa[0][var],bb[0][var],diff[var],rel[var],var,ind,case1,case2,\
                                        aa[1][var],bb[1][var],aa[3],aa[4],lon.values,lat.values],\
                                       kwargs={'path':path,'reg':region})
                p.start()
                processes.append(p)
            for process in processes:
                process.join()
    if vl != None:
        print('\nPlotting all the extra variables\n')
        aa=gather_data(path1,vlist,path1.strip().split('/')[-3],sv='y')
        bb=gather_data(path2,vlist,path2.strip().split('/')[-3],sv='y')
        aa[0].load()
        bb[0].load()
        aa[1].load()
        bb[1].load()
        lon = aa[5].round()
        lat = aa[6].round()
        print('Loaded data\n')
        diff = bb[0]-aa[0]
        rel = (diff/abs(aa[0]))*100
        processes=[]
        for ind in [0,1,2]:
            for var,unit in zip(vlist,vunits):
                p = mp.Process(target=get_map,args=[aa[0][var],bb[0][var],diff[var],rel[var],var,ind,case1,case2,\
                                aa[1][var],bb[1][var],aa[3],unit,lon.values,lat.values],\
                               kwargs={'path':path,'reg':region})
                p.start()
                processes.append(p)
            for process in processes:
                process.join()
    if tb != None:
        print('\nProducing all budget tables')
        for aer in aer_list[:]:
            processes=[]
            for ind in [0,1,2]:
                p = mp.Process(target=get_all_tables,args=[ind,aer,path1,path2,case1,case2,path,region,local,model])
                p.start()
                processes.append(p)
            for process in processes:
                process.join()

    if rf != None:
        print('\nProducing all forcing analysis')
        processes=[]
        for ind in ['ANN','JJA','DJF']:
            p = mp.Process(target=get_forcing_df,args=[path1,path2,case1,case2,path+'/set02RF'],\
                           kwargs={'season':ind,'mod':model,'scrip':sc})
            p.start()
            processes.append(p)
        for process in processes:
            process.join()
        
        
    finish = time.perf_counter()
    print(f'\nFinished in {round(finish-start, 2)} second(s)')    

