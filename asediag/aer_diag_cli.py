import time
import argparse
from itertools import product
import multiprocessing as mp
from pathlib import Path
import shutil
import importlib.resources as pkg_resources

from asediag.aerosol_diag_SEgrid import get_forcing_df, gather_ProfData, get_vert_profiles
from asediag.aerosol_diag_SEgrid import gather_data, get_map, get_local_profiles
from asediag.asediag_utils import get_html, get_plocal, html_template, setup_output_directory
from asediag.gen_budgets import GenAerosolBudgets
from asediag.gen_spatial_distr import GenSpatialData, gen_4panel_maps

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
    parser.add_argument("-sp", help="splot", action='store_true',default=None)
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
    splot = args.sp
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
        
    ## Setting output path and copying template frontend html
    path = setup_output_directory(outpath, case1, case2, region)
    ## copying the template content to out dir (i.e. outpath)
    package_name = 'asediag'
    with pkg_resources.open_text(package_name, 'aerosol_temp.html') as file:
        filedata = file.read()
        filedata = filedata.replace('Test_case',case2+'_'+region)
        filedata = filedata.replace('Control_case',case1+'_'+region)
    with open(path / 'aerosol.html','w') as file:
        file.write(filedata)
        
    aer_list = ['bc','so4','dst','mom','pom','ncl','soa','num']    
    start = time.perf_counter()

    if hp == None:
        mapPath = setup_output_directory(outpath, case1, case2, region, child='set02')
        pv_name = {'1000':'Surface concentration',
                   '850':'Concentration at 850 hPa',
                   '500':'Concentration at 500 hPa',
                   '250':'Concentration at 250 hPa'
                   }
        
        aer_list = ['bc','so4','dst','mom','pom','ncl','soa','num','DMS','SO2','H2SO4']

        if pv == None:
            html,title,tmp = get_html("season_latlon_bdn.png","Column-integrated burden")
            html_code = html_template(title,html,tmp)
            with open(mapPath / 'index_burden.html','w') as file:
                file.write(html_code)
        else:
            html,title,tmp = get_html("season_latlon_"+str(pv)+".png",pv_name[str(pv)])
            html_code = html_template(title,html,tmp)
            with open(mapPath / str('index_'+str(pv)+'.html'),'w') as file:
                file.write(html_code)

        for aer in aer_list[:]:
            print('getting data\n')
            print(path1,path2)
            print(path1.strip().split('/')[-3],pv)
            CntlMapInfo = GenSpatialData(path1,case1,aer,mod=model,plev=pv,reg=region,land=lnd,scrip=sc)
            TestMapInfo = GenSpatialData(path2,case2,aer,mod=model,plev=pv,reg=region,land=lnd,scrip=sc)
            aa = CntlMapInfo.gather_data()
            bb = TestMapInfo.gather_data()
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
            for ind,var in product([0,1,2],aa[2][:]):
                p = mp.Process(target=gen_4panel_maps,args=[aa[0][var],bb[0][var],diff[var],rel[var],var,ind,case1,case2,\
                                        aa[1][var],bb[1][var],aa[3],aa[4],lon,lat],\
                                       kwargs={'scrip':sc,'path':mapPath,'reg':region})
                p.start()
                processes.append(p)
            for process in processes:
                process.join()
    else:
        print('\nPlotting is muted\n')

    if vl != None:
        mapPath = setup_output_directory(outpath, case1, case2, region, child='set02')
        vlist = vl.split(',')
        vunits = vunit.split(',')
        assert len(vlist) == len(vunits), "List of variables and units should have the same length!"
        spfull_vars = ['AODVIS','AODBC','AODMODE1','BURDENBC','BURDEN1'] # Hard-coded for utility
        html,title,tmp = get_html("season_latlon_radiation.png","Other variables",listofvs=vlist,spfull_vars=spfull_vars,extra=[''])
        html_code = html_template(title,html,tmp)
        with open(mapPath / 'index_other.html','w') as file:
            file.write(html_code)
        print('getting data\n')
        print(path1,path2)
        print('\nPlotting all the extra variables\n')
        CntlMapInfo = GenSpatialData(path1,case1,vlist,mod=model,plev=pv,reg=region,land=lnd,scrip=sc,sv='y')
        TestMapInfo = GenSpatialData(path2,case2,vlist,mod=model,plev=pv,reg=region,land=lnd,scrip=sc,sv='y')
        aa = CntlMapInfo.gather_data()
        bb = TestMapInfo.gather_data()
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
        vlist = aa[2]
        processes=[]
        for ind,var in product([0,1,2],vlist[:]):
            unit = vunits[vlist.index(var)]
            p = mp.Process(target=gen_4panel_maps,args=[aa[0][var],bb[0][var],diff[var],rel[var],var,ind,case1,case2,\
                                        aa[1][var],bb[1][var],aa[3],aa[4],lon,lat],\
                                       kwargs={'scrip':sc,'path':mapPath,'reg':region})
            p.start()
            processes.append(p)
        for process in processes:
            process.join()

    if profile != None:
        sites,lats,lons = get_plocal(local)
        aer_list = ['bc','so4','dst','mom','pom','ncl','soa','num','DMS','SO2','H2SO4']
        html,title,tmp = get_html("season_lathgt.png","Vertical contour plots of zonal means",locations=sites)
        html_code = html_template(title,html,tmp)
        with open(path+'/set01/index_orig.html','w') as file:
            file.write(html_code)
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
        ExtraProfPath = setup_output_directory(outpath, case1, case2, region, child='set01')
        sites,lats,lons = get_plocal(local)
        eprof_list = extraprof.split(',')
        gunits = gunit.split(',')
        assert len(eprof_list) == len(gunits), "List of variables and units should have the same length!"
        spfull_vars = ['CCN3','dgnd_a01','dgnw_a01','bc_a1_sfgaex1','bc_a1_sfcoag1'] # Hard-coded for utility
        html,title,tmp = get_html("season_lathgt.png","Vertical contour plots of zonal means",locations=sites,listofvs=eprof_list,spfull_vars=spfull_vars,extra=[''])
        html_code = html_template(title,html,tmp)
        with open(ExtraProfPath / 'index_other.html','w') as file:
            file.write(html_code)
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
                get_local_profiles(d1[var],d2[var],ddiff[var],drel[var],var,case1,case2,sites[i],path=ExtraProfPath)

        processes=[]
        for ind,var in product([0,1,2],aa[1]):
            unit = gunits[eprof_list.index(var)]
            p = mp.Process(target=get_vert_profiles,args=[aa[0][var],bb[0][var],diff[var],rel[var],var,ind,case1,case2,\
                                    ],kwargs={'path':ExtraProfPath,'gunit':unit})
            p.start()
            processes.append(p)
        for process in processes:
            process.join()

    if (tb != None) and (splot == None):
        aer_list = ['bc','so4','dst','mom','pom','ncl','soa','num','DMS','SO2','H2SO4','SOAG']
        print('\nProducing all budget tables')
        html, title,tmp = get_html("season.html","Aerosol budget",locations=['Figure'],fmt='png')
        html_code = html_template(title,html,tmp)
        tabPath = setup_output_directory(outpath, case1, case2, region, child='tables')
        with open(tabPath / 'index.html','w') as file:
            file.write(html_code)
        for aer in aer_list[:]:
            processes=[]
            for ind in [0,1,2]:
                genBudget = GenAerosolBudgets(ind,aer,path1,path2,case1,case2,str(tabPath),region,local,model,lnd,splot)
                p = mp.Process(target=genBudget.generate_html_tables)
                p.start()
                processes.append(p)
            for process in processes:
                process.join()

    if splot != None:
        aer_list = ['bc','so4','pom','soa','num','DMS','SO2','H2SO4','SOAG']
        print('\nProducing all budget table figures')
        html, title,tmp = get_html("season.html","Aerosol budget",locations=['Figure'],fmt='png')
        html_code = html_template(title,html,tmp)
        tabPath = setup_output_directory(outpath, case1, case2, region, child='tables')
        with open(tabPath / 'index.html','w') as file:
            file.write(html_code)
        for aer,ind in product(aer_list[:],[0]):
            ss = ['ANN','DJF','JJA']
            print('\nTable variable plots for:',aer,ss[ind])
            genBudget = GenAerosolBudgets(ind,aer,path1,path2,case1,case2,str(tabPath),region,local,model,lnd,splot)
            columns = genBudget.generate_html_tables()
            d1, d2, diff, rel, vlist = genBudget.get_budget_SpData()
            d1.load()
            d2.load()
            diff.load()
            rel.load()
            processes=[]
            for col in columns:
                p = mp.Process(target=genBudget.gen_budget_spatialMaps,args=[d1,d2,diff,rel,vlist,col,sc])
                p.start()
                processes.append(p)
            for process in processes:
                process.join()

    if rf != None:
        html,title,tmp = get_html("season_latlon.html","Radiative forcings",listofvs=['Forcing'],spfull_vars=['Forcing'],locations=['TOA','SFC'],fmt='png')
        html_code = html_template(title,html,tmp)
        with open(path+'/set02RF/index.html','w') as file:
            file.write(html_code)
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

