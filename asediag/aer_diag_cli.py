import time
import argparse
from itertools import product
import multiprocessing as mp
from pathlib import Path
import pkg_resources
import shutil

from asediag.aerosol_diag_SEgrid import get_forcing_df, gather_ProfData, get_vert_profiles
from asediag.aerosol_diag_SEgrid import gather_data, get_map, get_all_tables, get_local_profiles
from asediag.asediag_utils import get_html, get_plocal

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-pval", help="pressure levels for latlon plots (options: 0, 200, 500 and 850)", default=None)
    parser.add_argument("-dir1", help="case 1 directory", default=None)
    parser.add_argument("-dir2", help="case 2 directory", default=None)
    parser.add_argument("-path", help="analysis output directory", default=Path('.').absolute())
    parser.add_argument("-cs1", help="case1 name", default=None)
    parser.add_argument("-cs2", help="case2 name", default=None)
    parser.add_argument("-m", help="model (options: eam & cam)", default='eam')
    parser.add_argument("-reg", help="select region", default=None)
    parser.add_argument("-land", help="select only land", action='store_true', default=None)
    parser.add_argument("-loc", help="select location", default=None)
    parser.add_argument("-vlist", help="plot extra variables defined in configuration", default=None)
    parser.add_argument("-vunit", help="units of the extra variables defined in configuration", default=None)
    parser.add_argument("-tab", help="get budget tables", action='store_true', default=None)
    parser.add_argument("-prof", help="get zonal mean vertical profile plots", action='store_true', default=None)
    parser.add_argument("-eprof", help="get zonal mean vertical profile plots for any variables", default=None)
    parser.add_argument("-gunit", help="units for the extra vertical profile plots", default=None)
    parser.add_argument("-forcing", help="get forcing analysis", action='store_true', default=None)
    parser.add_argument("-hplot", help="mute standard horizontal plots", action='store_true', default=None)
    parser.add_argument("-scrip", help="scrip file", \
                        default='/compyfs/www/hass877/share/emis_data/DECK120_to_SE/northamericax4v1pg2_scrip.nc')
   
    args = parser.parse_args()
    pv = args.pval
    path1 = args.dir1
    path2 = args.dir2
    outpath = args.path
    model = args.m
    region = args.reg
    lnd = args.land
    local = args.loc
    vl = args.vlist
    vunit = args.vunit
    tb = args.tab
    profile = args.prof
    extraprof = args.eprof
    gunit = args.gunit
    hp = args.hplot
    sc = args.scrip
    rf = args.forcing
    case1 = args.cs1
    case2 = args.cs2
    
    
    ## Add extra variables and units above here as necessaray
    if case1 == None:
        case1 = path1.strip().split('/')[-3]
    if case2 == None:
        case2 = path2.strip().split('/')[-3]
        
    ## This directories are moved earlier for batch submissions
    path = str(outpath)+'/'+case2+'_minus_'+case1
    print('\nSelected output directoy:',path)
    ## copying the template content to out dir (i.e. outpath)
    ## Getting rid of template directory would be best
    resource_package = __name__
    resource_path = 'template'
    tmp = pkg_resources.resource_filename(resource_package, resource_path)
    try:
        shutil.copytree(tmp, path)
    except FileExistsError:
        print('\nPath already exists!')
        pass
    except:
        print('\nCan not create directory:',tmp)
    # Rewriting the frontend here
    # !!This can be modified...Looks weird!!
    with open(tmp+'/aerosol.html','r') as file:
        filedata = file.read()
        filedata = filedata.replace('F20TR_v2_ndg_ERA5_SEdata_NA_RRM',case2)
        filedata = filedata.replace('F20TR_v2_ndg_ERA5_Defdata_NA_RRM',case1)
        if (region!=None) and (region!='Global') and (hp==None):
            Path(tmp+"/set02"+region).mkdir(exist_ok=True)
            filedata = filedata.replace('<p><span class="red">4.</span> <a href="set02RF/index.html">Radiative Forcing analysis</a></p>','<p><span class="red">4.</span> <a href="set02RF/index.html">Radiative Forcing analysis</a></p>\n<p><span class="red">5.</span> <a href="set02'+region+'/index.html">'+region+' Horizontal contour plots</a></p>')
    with open(path+'/aerosol.html','w') as file:
        file.write(filedata)
        
    aer_list = ['bc','so4','dst','mom','pom','ncl','soa','num']    
    start = time.perf_counter()
    # if __name__ == '__main__':
    if hp == None:
        aer_list = ['bc','so4','dst','mom','pom','ncl','soa','num','DMS','SO2','H2SO4']
        for aer in aer_list[:]:
            print('getting data\n')
            print(path1,path2)
            print(path1.strip().split('/')[-3],pv)
            aa=gather_data(path1,aer,case1,model,plev=pv,reg=region,land=lnd)
            bb=gather_data(path2,aer,case2,model,plev=pv,reg=region,land=lnd)
            aa[0].load()
            bb[0].load()
            aa[1].load()
            bb[1].load()
            lon = aa[5].round()
            lat = aa[6].round()
            if 'ncol' in aa[0].dims:
                lon = lon.values
                lat = lat.values
            print('Loaded data\n')
            diff = bb[0]-aa[0]
            rel = (diff/abs(aa[0]))*100
            processes=[]
            for ind,var in product([0,1,2],aa[2]):
                p = mp.Process(target=get_map,args=[aa[0][var],bb[0][var],diff[var],rel[var],var,ind,case1,case2,\
                                        aa[1][var],bb[1][var],aa[3],aa[4],lon,lat],\
                                       kwargs={'scrip':sc,'path':path+'/set02','reg':region})
                p.start()
                processes.append(p)
            for process in processes:
                process.join()
    else:
        print('\nPlotting is muted\n')

    if vl != None:
        vlist = vl.split(',')
        vunits = vunit.split(',')
        assert len(vlist) == len(vunits), "List of variables and units should have the same length!"
        print('getting data\n')
        print(path1,path2)
        print('\nPlotting all the extra variables\n')
        aa=gather_data(path1,vlist,case1,model,sv='y')
        bb=gather_data(path2,vlist,case2,model,sv='y')
        aa[0].load()
        bb[0].load()
        aa[1].load()
        bb[1].load()
        lon = aa[5].round()
        lat = aa[6].round()
        print('Loaded data\n')
        diff = bb[0]-aa[0]
        rel = (diff/abs(aa[0]))*100
        vlist = aa[2]
        processes=[]
        for ind,var in product([0,1,2],vlist):
            unit = vunits[vlist.index(var)]
            get_map(aa[0][var],bb[0][var],diff[var],rel[var],var,ind,case1,case2,\
                    aa[1][var],bb[1][var],aa[3],unit,lon,lat,\
                    scrip=sc,path=path+'/set02',reg=region)

    if profile != None:
        sites,lats,lons = get_plocal(local)
        aer_list = ['bc','so4','dst','mom','pom','ncl','soa','num','DMS','SO2','H2SO4']
        html = get_html("season_lathgt.png","Vertical contour plots of zonal means",locations=sites)
        with open(path+'/set01/index.html','w') as file:
            file.write(html)
        for aer in aer_list[:]:
            print('getting data\n')
            print(path1,path2)
            print('\nProducing profiles can take some time in SE-grid\nbinning data . . .\n')
            aa=gather_ProfData(path1,aer,case1,model,lats=lats,lons=lons)
            bb=gather_ProfData(path2,aer,case2,model,lats=lats,lons=lons)
            aa[0].load()
            bb[0].load()
            print('Loaded data\n')
            diff = bb[0]-aa[0]
            rel = (diff/abs(aa[0]))*100
            
            for i in range(len(sites)):
                d1 = aa[2].isel(location=i)
                d2 = bb[2].isel(location=i)
                ddiff = d2 - d1
                drel = (ddiff/abs(d1))*100
                for var in aa[1]:
                    get_local_profiles(d1[var],d2[var],ddiff[var],drel[var],var,case1,case2,sites[i],path=path+'/set01')

            processes=[]
            for ind,var in product([0,1,2],aa[1]):
                p = mp.Process(target=get_vert_profiles,args=[aa[0][var],bb[0][var],diff[var],rel[var],var,ind,case1,case2,\
                                        ],kwargs={'path':path+'/set01'})
                p.start()
                processes.append(p)
            for process in processes:
                process.join()
                
    if extraprof != None:
        sites,lats,lons = get_plocal(local)
        eprof_list = extraprof.split(',')
        gunits = gunit.split(',')
        assert len(eprof_list) == len(gunits), "List of variables and units should have the same length!"
        html = get_html("season_lathgt.png","Vertical contour plots of zonal means",extra=eprof_list,locations=sites)
        with open(path+'/set01/index.html','w') as file:
            file.write(html)
        print('getting data\n')
        print(path1,path2)
        print('\nProducing profiles can take some time in SE-grid\nbinning data . . .\n')
        aa=gather_ProfData(path1,eprof_list,case1,model,sv=True,lats=lats,lons=lons)
        bb=gather_ProfData(path2,eprof_list,case2,model,sv=True,lats=lats,lons=lons)
        aa[0].load()
        bb[0].load()
        print('Loaded data\n')
        diff = bb[0]-aa[0]
        rel = (diff/abs(aa[0]))*100

        for i in range(len(sites)):
            d1 = aa[2].isel(location=i)
            d2 = bb[2].isel(location=i)
            ddiff = d2 - d1
            drel = (ddiff/abs(d1))*100
            for var in aa[1]:
                get_local_profiles(d1[var],d2[var],ddiff[var],drel[var],var,case1,case2,sites[i],path=path+'/set01')

        processes=[]
        for ind,var in product([0,1,2],aa[1]):
            unit = gunits[eprof_list.index(var)]
            p = mp.Process(target=get_vert_profiles,args=[aa[0][var],bb[0][var],diff[var],rel[var],var,ind,case1,case2,\
                                    ],kwargs={'path':path+'/set01','gunit':unit})
            p.start()
            processes.append(p)
        for process in processes:
            process.join()

    if tb != None:
        aer_list = ['bc','so4','dst','mom','pom','ncl','soa','num','DMS','SO2','H2SO4']
        print('\nProducing all budget tables')
        html = get_html("season.html","Aerosol budget")
        with open(path+'/tables/index.html','w') as file:
            file.write(html)
        for aer in aer_list[:]:
            processes=[]
            for ind in [0,1,2]:
                p = mp.Process(target=get_all_tables,args=[ind,aer,path1,path2,case1,case2,path+'/tables',region,local,model,lnd])
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

