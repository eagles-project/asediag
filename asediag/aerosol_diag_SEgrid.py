import xarray as xr
import numpy as np
## imports for plots
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cartopy.crs as crs
from asediag.nclCols import amwg256_map, BlueWhiteOrangeRed_map
from asediag.aerdiag_plots import get_plots
from asediag.asediag_utils import rounding, get_html_table, get_vertint
from asediag.asediag_utils import gen_colbar_range, get_local, get_plocal, get_nearestlatlon
import matplotlib
import fnmatch
from pathlib import Path
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
pal = sns.color_palette("pastel",10)
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

##########################################################################
##########################################################################    
def get_hplots(path,case,ts,aer,plev=None,mod='eam',reg=None,land=None):
    ## reading data as xarray
    try:
        data = xr.open_mfdataset(path+case+'.'+mod+'.'+ts+'.*_climo.nc')
        lon = data['lon']
        lon[lon > 180.] -= 360.
    except:
        data = xr.open_mfdataset(path+case+'*'+ts+'*_climo.nc').isel(time=0)
        lon = xr.where(data.lon > 180,data.lon-360,data.lon)
        lon = lon.assign_coords(lon=lon.values)
        data['lon'] = lon
        lon = lon.sortby(lon)
        data = data.sortby('lon')
    lat = data['lat']  
    if reg!=None:
        lat1,lat2,lon1,lon2=get_latlon(reg)
    else:
        lat1,lat2,lon1,lon2=lat.values.min(),lat.values.max(),lon.values.min(),lon.values.max()
    
    
    if 'year' in data.coords:
        data = data.rename({'year':'season'})
    pval = 'bdn'
    fact = 1e9
    factaa = 1.01325e5 / 8.31446261815324 / 273.15 * 28.9647 / 1.e9   # kg-air/cm3-air
    factbb = factaa * 1.e15  # ug-air/m3-air
    if plev == '850':
        tlev=data.lev[(data.lev.values>845) & (data.lev.values<855)].values
        data = data.sel(lev=tlev)
        data = data.isel(lev=0)
        pval = '850'
        fact = factbb
    elif plev == '500':
        tlev=data.lev[(data.lev.values>490) & (data.lev.values<505)].values
        data = data.sel(lev=tlev)
        data = data.isel(lev=0)
        pval = '500'
        fact = factbb
    elif plev == '200':
        tlev=data.lev[(data.lev.values>195) & (data.lev.values<205)].values
        data = data.sel(lev=tlev)
        data = data.isel(lev=0)
        pval = '200'
        fact = factbb
    elif plev == '0':
        tlev=data.lev[-1].values
        data = data.sel(lev=tlev)
        pval = '1013'
        fact = factbb
    if aer=='num':
        if plev!=None:
            fact = factaa
        else:
            fact = 1.0
    ## factors
    grav = 9.806
    ha = data['hyai']
    hb = data['hybi']
    p0 = data['P0']
    ps = data['PS']
    area = data['area']
    landF = data['LANDFRAC']
    ## all variable list
    vlist = list(data.variables.keys())
    # Total BC burden
    gvars = ['SO2','DMS','H2SO4']
    if aer in gvars:
        var_avars = fnmatch.filter(vlist,aer)
        var_cvars = []
    else:
        var_avars = fnmatch.filter(vlist,aer+'_a?')
        var_cvars = fnmatch.filter(vlist,aer+'_c?')
    var_vars = var_avars+var_cvars
    print(var_vars)
    vdata = data[var_vars]
    if plev == None:
        ## calc. dp
        vdata = get_vertint(vdata,ha,p0,hb,ps,grav,fact)
    else:
        vdata = vdata*fact
    if land==True:
        vdata = vdata.where(landF>0)
    else:
        vdata = vdata.where(landF>=0)
    ## getting total
    vdata[aer] = vdata.to_array().sum('variable')
    ## actual mean
    vdatalatlon = vdata.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
    arealatlon = area.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
    mean = (vdatalatlon*arealatlon).sum(vdatalatlon.dims)/(arealatlon).sum(arealatlon.dims)

    return vdata,mean,var_vars+[aer],pval,lon,lat

def get_vplots(path,case,ts,aer,mod='eam',lats=None,lons=None):
    ## reading data as xarray
    try:
        data = xr.open_mfdataset(path+case+'.'+mod+'.'+ts+'.*_climo.nc')
        lon = data['lon']
        lon[lon > 180.] -= 360.
    except:
        data = xr.open_mfdataset(path+case+'*'+ts+'*_climo.nc').isel(time=0)
        lon = xr.where(data.lon > 180,data.lon-360,data.lon)
        lon = lon.assign_coords(lon=lon.values)
        data['lon'] = lon
        lon = lon.sortby(lon)
        data = data.sortby('lon')
    lat = data['lat']
            
    fact = 1e9
    factaa = 1.01325e5 / 8.31446261815324 / 273.15 * 28.9647 / 1.e9   # kg-air/cm3-air
    factbb = factaa * 1.e15  # ug-air/m3-air
    if 'year' in data.coords:
        data = data.rename({'year':'season'})
    if aer=='num':
        fact = factaa
    else:
        fact = factbb
    ## all variable list
    vlist = list(data.variables.keys())
    ## get list of available variables
    gvars = ['SO2','DMS','H2SO4']
    if aer in gvars:
        var_avars = fnmatch.filter(vlist,aer)
        var_cvars = []
    else:
        var_avars = fnmatch.filter(vlist,aer+'_a?')
        var_cvars = fnmatch.filter(vlist,aer+'_c?')
    var_vars = var_avars+var_cvars
    print(var_vars)
    vdata = data[var_vars]
    vdata = vdata*fact
    ## getting total
    vdata[aer] = vdata.to_array().sum('variable')
    
    lts = []
    lns = []
    for lat1,lon1 in zip(lats,lons):
        lt1,lt2,ln1,ln2 = get_nearestlatlon(lon1,lat1,lon,lat)
        lts.append(lt1)
        lns.append(ln1)
    pdata = []
    for ln,lt in zip(lns,lts):
        var1 = vdata.where((lat==lt) & (lon==ln)).copy()
        var1 = var1.stack(grid=var1.dims)
        var1 = var1.dropna("grid", how="all")
        levels = var1.lev.values
        var1 = var1.drop_vars('lev')
        var1 = var1.assign_coords(grid=levels)
        pdata.append(var1)
    local_data = xr.concat(pdata,dim="location")
    
    if 'ncol' in data.dims:
        ll=data['lat'].round().values.tolist()
        all_ll=ll*len(data.lev)
        dd=vdata.to_dataframe()
        dd=dd.drop(columns=['season'])
        dd['lat']=all_ll
        dd=dd.groupby(['lev','lat']).mean()
        ## resampling to lower res (2 deg)
        dd = dd.rolling(2).mean()
        dd = dd.iloc[::2,:]
        vdata=dd.to_xarray()
    else:
        vdata = vdata.mean(dim='lon')
    return vdata,var_vars+[aer],local_data

def get_svplots(path,case,ts,var_vars,mod='eam',lats=None,lons=None):
    ## reading data as xarray
    try:
        data = xr.open_mfdataset(path+case+'.'+mod+'.'+ts+'.*_climo.nc')
        lon = data['lon']
        lon[lon > 180.] -= 360.
    except:
        data = xr.open_mfdataset(path+case+'*'+ts+'*_climo.nc').isel(time=0)
        lon = xr.where(data.lon > 180,data.lon-360,data.lon)
        lon = lon.assign_coords(lon=lon.values)
        data['lon'] = lon
        lon = lon.sortby(lon)
        data = data.sortby('lon')
    lat = data['lat']
    fact = 1
    if 'year' in data.coords:
        data = data.rename({'year':'season'})
    print('Extra variables:',var_vars)
    vdata = data[var_vars]
    vdata = vdata*fact

    lts = []
    lns = []
    for lat1,lon1 in zip(lats,lons):
        lt1,lt2,ln1,ln2 = get_nearestlatlon(lon1,lat1,lon,lat)
        lts.append(lt1)
        lns.append(ln1)
    pdata = []
    for ln,lt in zip(lns,lts):
        var1 = vdata.where((lat==lt) & (lon==ln)).copy()
        var1 = var1.stack(grid=var1.dims)
        var1 = var1.dropna("grid", how="all")
        levels = var1.lev.values
        var1 = var1.drop_vars('lev')
        var1 = var1.assign_coords(grid=levels)
        pdata.append(var1)
    local_data = xr.concat(pdata,dim="location")

    ## getting total
    if 'ncol' in data.dims:
        ll=data['lat'].round().values.tolist()
        all_ll=ll*len(data.lev)
        dd=vdata.to_dataframe()
        dd=dd.drop(columns=['season'])
        dd['lat']=all_ll
        dd=dd.groupby(['lev','lat']).mean()
        ## resampling to lower res (2 deg)
        dd = dd.rolling(2).mean()
        dd = dd.iloc[::2,:]
        vdata=dd.to_xarray()
    else:
        vdata = vdata.mean(dim='lon')
    return vdata,var_vars,local_data

def get_singleV_hplots(path,case,ts,var,fact=1,vertinit=None,pval='radiation',mod='eam'):
    ## reading data as xarray
    try:
        data = xr.open_mfdataset(path+case+'.'+mod+'.'+ts+'.*_climo.nc')
        lon = data['lon']
        lon[lon > 180.] -= 360.
    except:
        data = xr.open_mfdataset(path+case+'*'+ts+'*_climo.nc').isel(time=0)
        lon = xr.where(data.lon > 180,data.lon-360,data.lon)
        lon = lon.assign_coords(lon=lon.values)
        data['lon'] = lon
        lon = lon.sortby(lon)
        data = data.sortby('lon')
    lat = data['lat']
    if ts=='ANN':
        data = data.rename({'year':'season'})   
    ## factors
    grav = 9.806
    ha = data['hyai']
    hb = data['hybi']
    p0 = data['P0']
    ps = data['PS']
    area = data['area']
    ## select var
    vdata = data[var]
    if vertinit != None:
        ## calc. dp
        vdata = get_vertint(vdata,ha,p0,hb,ps,grav,fact)
    else:
        vdata = vdata*fact
    ## actual mean
    mean = (vdata*area).sum(area.dims)/(area).sum(area.dims)
    return vdata,mean,var,pval,lon,lat

def get_tables(path,case,ts,aer,reg=None,loc=None,mod='eam',indl=None,land=False,splot=None):
    try:
        data = xr.open_mfdataset(path+case+'.'+mod+'.'+ts+'.*_climo.nc')
        lon = data['lon'].values
        lon[lon > 180.] -= 360.
    except:
        data = xr.open_mfdataset(path+case+'*'+ts+'*_climo.nc').isel(time=0)
        lon = xr.where(data.lon > 180,data.lon-360,data.lon)
        lon = lon.assign_coords(lon=lon.values)
        data['lon'] = lon
        lon = lon.sortby(lon)
        data = data.sortby('lon')
        indl = None
    lat = data['lat']
    if reg!=None:
        lat1,lat2,lon1,lon2=get_latlon(reg)
    elif loc!=None:
        lat1,lon1=get_local(loc)
        lat1,lat2,lon1,lon2 = get_nearestlatlon(lon1,lat1,lon,lat)
    else:
        lat1,lat2,lon1,lon2=lat.values.min(),lat.values.max(),lon.min(),lon.max()
    if 'year' in data.coords:
        data = data.rename({'year':'season'})
    ## factors
    fact = 1e-9
    grav = 9.806
    ha = data['hyai']
    hb = data['hybi']
    p0 = data['P0']
    ps = data['PS']
    area = data['area']*(6.37122e6)**2
    landF = data['LANDFRAC']
    avgod = 6.022e+23
    mwso4 = 115.0
    factaa  = mwso4*10.0/avgod        # convert molec/cm2/s to kg/m2/s
    factaaa = 12.0*10.0/avgod         # convert molec/cm2/s to kg/m2/s
    factbb  = 86400.0*365.0*1e-9      # convert kg/s to Tg/yr
    if aer == 'num':  
        factbb  = 86400.0*365.0
        factaa = 1.e4/(avgod*1.e3)
    factcc  = factbb/mwso4*32.066     # convert kg/s to TgS/yr
    factdd  = 32.066/mwso4*1e-9       # convert kg to TgS
    sum_airmass = ((ps*area).sum()/grav)*1e6
    fact_kgpkg_kgpcm3 = 1.01325e5 / 8.31446261815324 / 273.15 * 28.9647 / 1.e9   # kg-air/cm3-air
    fact_kgpcm3_ugpm3 = fact_kgpkg_kgpcm3 * 1e15
    
    ## all variable list
    vlist = list(data.variables.keys())
    # List of all variables considered
    avariables = [aer+'_a?',aer+'_a?'+'DDF',aer+'_a?'+'SFWET','SF'+aer+'_a?',aer+'_a?'+'_CLXF',\
     aer+'_a?'+'_sfgaex1',aer+'_a?'+'GVF',aer+'_a?'+'TBF',aer+'_a?'+'SFSIS',\
     aer+'_a?'+'SFSIC',aer+'_a?'+'SFSBS',aer+'_a?'+'SFSBC',aer+'_a?'+'SFSES',\
     aer+'_a?'+'SFSEC',aer+'_a?'+'_sfgaex2',aer+'_a?'+'_sfcoag1',aer+'_a?'+'_sfcsiz3',\
     aer+'_a?'+'_sfcsiz4',aer+'_a?'+'_mixnuc1',aer+'_a?'+'AQH2SO4',\
     aer+'_a?'+'AQSO4',aer+'_a?'+'_sfnnuc1','AQ_'+aer+'_a?','GS_'+aer+'_a?',aer+'_a?']
    cvariables = [aer+'_c?',aer+'_c?'+'DDF',aer+'_c?'+'SFWET','SF'+aer+'_c?',aer+'_c?'+'_CLXF',\
     aer+'_c?'+'_sfgaex1',aer+'_c?'+'GVF',aer+'_c?'+'TBF',aer+'_c?'+'SFSIS',\
     aer+'_c?'+'SFSIC',aer+'_c?'+'SFSBS',aer+'_c?'+'SFSBC',aer+'_c?'+'SFSES',\
     aer+'_c?'+'SFSEC',aer+'_c?'+'_sfgaex2',aer+'_c?'+'_sfcoag1',aer+'_c?'+'_sfcsiz3',\
     aer+'_c?'+'_sfcsiz4',aer+'_c?'+'_mixnuc1',aer+'_c?'+'AQH2SO4',\
     aer+'_c?'+'AQSO4',aer+'_c?'+'_sfnnuc1','AQ_'+aer+'_c?','GS_'+aer+'_c?',aer+'_c?']
    
    ## handle gas=phase species
    gvars = ['SO2','DMS','H2SO4']
    if aer in gvars:
        avariables = str(avariables).replace('_a?','').replace("'",'').replace(' ','')[1:-1].replace(aer+'DDF','DF_'+aer).replace(aer+'SFWET','WD_'+aer).split(',')
        cvariables = ['']*len(cvariables)
    
    # sfc emis
    df = pd.DataFrame()
    nvar=0
    for avar,cvar in zip(avariables[:],cvariables[:]):
        nvar+=1
        var_avars = fnmatch.filter(vlist,avar)
        var_cvars = fnmatch.filter(vlist,cvar)
        var_vars = var_avars+var_cvars

        vdata = data[var_vars]
        if ((avar == aer+'_a?') and (nvar == 1)) or ((avar == aer) and (nvar == 1)):
            vars1 = var_vars+[avar+'+'+cvar]
            if (aer == 'so4') or (aer in gvars):
                bname = 'Burden (TgS)'
                srcname = 'Sources (TgS/yr)'
                snkname = 'Sinks (TgS/yr)'
                vdata = get_vertint(vdata,ha,p0,hb,ps,grav,factdd)
            elif aer == 'num':
                bname = 'Burden (#/mg-air)'
                srcname = 'Sources (#/mg-air/yr)'
                snkname = 'Sinks (#/mg-air/yr)'
                vdata = get_vertint(vdata,ha,p0,hb,ps,grav,1)/sum_airmass
            else:
                bname = 'Burden (Tg)'
                srcname = 'Sources (Tg/yr)'
                snkname = 'Sinks (Tg/yr)'
                vdata = get_vertint(vdata,ha,p0,hb,ps,grav,fact)
        elif ((avar == aer+'_a?') and (nvar > 1)) or ((avar == aer) and (nvar > 1)):
            if (aer == 'so4') or (aer in gvars):
                #sfc conc
                sname = 'Sfc Conc. (ug/m3)'
                vdata = vdata[dict(lev=-1)].drop_vars('lev')
                vdata = vdata*fact_kgpcm3_ugpm3
            elif aer == 'num':
                #sfc conc
                sname = 'Sfc Conc. (#/cm3)'
                vdata = vdata[dict(lev=-1)].drop_vars('lev')
                vdata = vdata*1
            else:
                #sfc conc
                sname = 'Sfc Conc. (ug/m3)'
                vdata = vdata[dict(lev=-1)].drop_vars('lev')
                vdata = vdata*fact_kgpcm3_ugpm3
        else:
            if ('_CLXF' in avar):
                if (aer == 'bc') or (aer == 'pom'):
                    vdata = vdata*factaaa*factbb
                elif aer == 'num':
                    vdata = vdata*factaa*factbb/sum_airmass
                elif (aer == 'so4') or (aer in gvars):
                    vdata = vdata*factaa*factcc
                else:
                    vdata = vdata*factaa
            else:
                if (aer == 'so4') or (aer in gvars):
                    if ('ncol' in data.dims) and (len(vdata.dims) > 1):
                        vdata = get_vertint(vdata,ha,p0,hb,ps,grav,factcc)
                    elif len(vdata.dims) > 2:
                        vdata = get_vertint(vdata,ha,p0,hb,ps,grav,factcc)
                    else:
                        vdata = vdata*factcc
                elif aer == 'num':
                    if ('ncol' in data.dims) and (len(vdata.dims) > 1):
                        vdata = get_vertint(vdata,ha,p0,hb,ps,grav,factbb)/sum_airmass
                    elif len(vdata.dims) > 2:
                        vdata = get_vertint(vdata,ha,p0,hb,ps,grav,factbb)/sum_airmass
                    else:
                        vdata = vdata*factbb/sum_airmass
                else:
                    if ('ncol' in data.dims) and (len(vdata.dims) > 1):
                        vdata = get_vertint(vdata,ha,p0,hb,ps,grav,factbb)
                    elif len(vdata.dims) > 2:
                        vdata = get_vertint(vdata,ha,p0,hb,ps,grav,factbb)
                    else:
                        vdata = vdata*factbb
        ## getting total
        prob_list=[]
        for item in vars1[:-1]:
            prob_list.append(avar.replace(aer+'_a?',item))
        unavail_vars = list(set(prob_list)-set(var_vars+[avar+'+'+cvar]))
        if var_vars!=[]:
            vdata[avar+'+'+cvar] = vdata.to_array().sum('variable')
        else:
            vdata[avar+'+'+cvar] = np.nan
        if unavail_vars!=[]:
            vdata[unavail_vars] = [np.nan]*len(unavail_vars)

        if land==True:
            vdata = vdata.where(landF>0)
        else:
            vdata = vdata.where(landF>=0)
    
        if indl != None:
            try:
                mean = (vdata.sel(ncol=indl)).mean(dim=['ncol'])
            except:
                vdatalatlon = vdata.where((lon>=lon1) & (lon<=lon2))
                vdatalatlon = vdatalatlon.where((lat>=lat1) & (lat<=lat2))
                arealatlon = area.where((lon>=lon1) & (lon<=lon2))
                arealatlon = arealatlon.where((lat>=lat1) & (lat<=lat2))
                mean = (vdatalatlon*arealatlon).sum(arealatlon.dims)
        elif ('WD_' in avar):
            vdatalatlon = vdata.where((lon>=lon1) & (lon<=lon2))
            vdatalatlon = vdatalatlon.where((lat>=lat1) & (lat<=lat2))
            mean = (vdatalatlon).sum(arealatlon.dims)
        else:
            vdatalatlon = vdata.where((lon>=lon1) & (lon<=lon2))
            vdatalatlon = vdatalatlon.where((lat>=lat1) & (lat<=lat2))
            arealatlon = area.where((lon>=lon1) & (lon<=lon2))
            arealatlon = arealatlon.where((lat>=lat1) & (lat<=lat2))
            mean = (vdatalatlon*arealatlon).sum(arealatlon.dims)
            
        ## Renaming available variables
        rvars = dict(zip(prob_list+[avar+'+'+cvar],vars1))
        mean = mean.rename_vars(rvars)
        
        # Treat specific varaibles
        if (('DDF' in avar) or ('GVF' in avar) or ('TBF' in avar) or ('DF_' in avar) or ('WD_' in avar)):
            mean = -1*mean
        
        ### Treat SO2 separately: Based on discussions with Jianfeng Li (jianfeng.li@pnnl.gov)
        if aer == 'SO2':
            if 'WD_' in avar:
                wdep_so2 = mean
            elif '_CLXF' in avar:
                elev_emis_so2 = mean
            # GS_SO2 = SO2 emission (elevated) + WD_SO2 + Chemical reactions
            elif 'GS_' in avar:
                mean = mean + wdep_so2 - elev_emis_so2
        
        ## Appending to dataframe
        ndf = mean.expand_dims(dim='vars').to_dataframe()
        df = pd.concat([df,ndf.replace(0, np.nan)])
        
    if 'ncol' in data.dims:    
        df['year'] = np.nan
        df = df[['season','year']+vars1]
    else:
        df['time'] = np.nan
        df = df[['time']+vars1]
        
    index_list = [bname,'Dry deposition','Wet deposition','surface emission',\
             'elevated emission','condensation-aging','gravitational','turbulent',\
             'incloud, stratiform','incloud, convective','belowcloud, strat.',\
              'belowcloud, convec.','rain evap, strat.','rain evap, convec.',\
             'renaming (sfgaex2)','coagulation (sfcoag1)','calcsize (sfcsiz3)',\
             'calcsize (sfcsiz4)','dropmixnuc (mixnuc1)','cloudchem (AQH2SO4)',\
             'cloudchem (AQSO4)','NPF (sfnnuc1)','Aq. chem (gas-species)','gas chem/wet dep. (gas-species)',sname]
    df.index=index_list
    listofSS = ['Dry deposition','Wet deposition','renaming (sfgaex2)',\
                 'coagulation (sfcoag1)','calcsize (sfcsiz3)',\
                 'calcsize (sfcsiz4)','dropmixnuc (mixnuc1)',\
                 'condensation-aging','surface emission','elevated emission',\
                 'cloudchem (AQH2SO4)','cloudchem (AQSO4)','NPF (sfnnuc1)',\
                 'Aq. chem (gas-species)','gas chem/wet dep. (gas-species)']
    if aer in gvars:
        aer = 'total_'+aer
        df.columns=df.columns.tolist()[:-1]+[aer]
        srcsnk = df.loc[listofSS][vars1[:-1]+[aer]]
    else:
        df.columns=df.columns.tolist()[:-1]+[aer]
        srcsnk = df.loc[listofSS[:-2]][vars1[:-1]+[aer]]
    
    ## Estimating sources and sinks
    src = srcsnk.where(srcsnk>0).sum()
    if 'SO2' in aer:
        snk = df.loc['gas chem/wet dep. (gas-species)'] + df.loc['Dry deposition']
    else:
        snk = srcsnk.where(srcsnk<0).sum()
    df.loc[srcname] = src
    df.loc[snkname] = snk
    lifetime = (df.loc[bname][vars1[:-1]+[aer]]/abs(df.loc[snkname][vars1[:-1]+[aer]]))*365
    df.loc['Lifetime (days)'] = lifetime
    if 'ncol' in data.dims:
        df['season']=ts
    else:
        df['time']=ts
    df = df.reindex([bname,sname,srcname,'surface emission','elevated emission',snkname,\
               'Dry deposition','gravitational','turbulent','Wet deposition',\
               'incloud, stratiform','incloud, convective','belowcloud, strat.',\
               'belowcloud, convec.','rain evap, strat.','rain evap, convec.',\
               'Lifetime (days)','renaming (sfgaex2)','coagulation (sfcoag1)','calcsize (sfcsiz3)',\
                'calcsize (sfcsiz4)','dropmixnuc (mixnuc1)','cloudchem (AQH2SO4)',\
                'cloudchem (AQSO4)','condensation-aging','NPF (sfnnuc1)','Aq. chem (gas-species)','gas chem/wet dep. (gas-species)'])
    return df

def get_budget_plot(data,path,var,ind,case1,case2,unit='(#/mg-air/yr)'):
    data = data.reset_index()
    data.columns = ['metric','cntl','test','diff','rel']
    data = data.drop(index=[7,8,10,11,12,13,14,15,16,19,20,27]).reset_index(drop=True)
    xx = ['Burden','Sfc Conc.','Total\nSource','Surface\nemission','Elevated\nemission','Total\nSink', \
          'Dry\ndeposition', 'Wet\ndeposition','Renaming','Coagulation','Dropmix\nnuc',\
          'Cloud\nchemistry\n(AQH2SO4)', 'Cloud\nchemistry\n(AQSO4)','Condensation','NPF',\
          'Aquous\nchemistry\n(gas-species)']
    data['metric'] = xx
    ## Get sinks
    data_sink = data[data['cntl']<0].sort_values(by='diff',ascending=False).reset_index(drop=True)
    selrel = (data_sink['diff']/abs(data_sink['diff']).max())*100
    data_sink = data_sink[abs(selrel)>=1].reset_index(drop=True)
    data_sink['diff'] = np.sign(data_sink['cntl'])*data_sink['diff']
    ## Get sources
    data_source = data.drop(index=[0,1])[data['cntl']>0].sort_values(by='diff',ascending=True).reset_index(drop=True)
    selrel = (data_source['diff']/abs(data_source['diff']).max())*100
    data_source = data_source[abs(selrel)>=1].reset_index(drop=True)
    data_source['diff'] = np.sign(data_source['cntl'])*data_source['diff']

    ## Plot Figure
    fig=plt.figure(figsize=[18,10])
    gs = gridspec.GridSpec(1, 5,wspace=0.2)
    ax1 = fig.add_subplot(gs[:, :4])
    xx = data_source['metric'].tolist()+data_sink['metric'].tolist()
    hh=0.2
    ax1.bar(np.arange(0, len(data_source)), (data_source['diff']),color=pal[0],edgecolor='k',zorder=4,label='Sources')
    ax1.bar(np.arange(len(data_source), len(data_source)+len(data_sink)), (data_sink['diff']),color=pal[1],edgecolor='k',zorder=4,label='Sinks')
    ax1.set_xticks(np.arange(0, len(data_source)+len(data_sink)),xx)
    ax1.grid(linestyle='--',color='#EBE7E0',zorder=4)
    ax1.set_axisbelow(True)
    plt.ylabel('$\Delta$('+var+')\n'+unit,fontsize=20)
    plt.xlabel('')
    plt.tick_params(labelsize=15)
    plt.setp(ax1.spines.values(),lw=1.5)
    ax1.tick_params(axis='x',which='both',bottom=False)
    plt.axhline(0,c='k')
    plt.legend(fontsize=15)

    ## second axis
    ax2 = fig.add_subplot(gs[:, 4:])
    ax2.bar(0, data.iloc[0]['rel'],color=pal[2],width=0.5,edgecolor='k',zorder=4)
    ax2.bar(1, data.iloc[1]['rel'],color=pal[2],width=0.5,edgecolor='k',zorder=4)
    ax2.set_xticks([-0.5,0,1,1.5],['','Burden','Surface\nconc.',''])
    ax2.grid(linestyle='--',color='#EBE7E0',zorder=4)
    ax2.set_axisbelow(True)
    plt.ylabel('Percent change (%)',fontsize=20)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    plt.xlabel('')
    plt.tick_params(labelsize=15)
    plt.setp(ax2.spines.values(),lw=1.5)
    ax2.tick_params(axis='x',which='both',bottom=False)
    plt.axhline(0,c='k')
    
    fig.suptitle(r'$\bf{Control\ Case:}$ '+case1+'\n'+\
                 r'$\bf{Test\ Case:}$ '+case2+'\n'+r'$\bf{Plotting:}$ '+var,\
                 fontsize=20,horizontalalignment='left',x=0.125,y=0.98)
    ## Saving figure
    ss = ['ANN','DJF','JJA']
    plt.savefig(str(path)+'/'+var+'_Figure.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
    plt.close()


def get_all_tables(ind,aer,path1,path2,case1,case2,path,reg,loc,mod,land,splot):
    ss = ['ANN','DJF','JJA']
    cdatadef=get_tables(path1,case1,ss[ind],aer,reg=reg,loc=loc,mod=mod,land=land,splot=splot)
    cdatase=get_tables(path2,case2,ss[ind],aer,reg=reg,loc=loc,mod=mod,land=land,splot=splot)
    if 'year' in cdatadef.columns:
        cdatadef = cdatadef.drop('year', axis=1)
    if 'year' in cdatase.columns:
        cdatase = cdatase.drop('year', axis=1)
    cdatadiff = cdatase[cdatase.columns[1:]] - cdatadef[cdatase.columns[1:]]
    cdatarel = (cdatadiff/abs(cdatadef[cdatase.columns[1:]]))*100
    for col in cdatarel.columns:
        df = pd.DataFrame()
        df['<a target="_blank" href="{}">Control Case</a>'.format(col+'_'+case1+'_'+ss[ind]+'_latlon_splots.png')]=cdatadef[col]
        df['<a target="_blank"  href="{}">Test Case</a>'.format(col+'_'+case2+'_'+ss[ind]+'_latlon_splots.png')]=cdatase[col]
        df['<a target="_blank" href="{}">difference</a>'.format(col+'_diff_'+ss[ind]+'_latlon_splots.png')]=cdatadiff[col]
        df['<a target="_blank" href="{}">rel diff (%)</a>'.format(col+'_rel_'+ss[ind]+'_latlon_splots.png')]=cdatarel[col]
        if ind == 0:
            if 'num' in aer:
                gunit = '(#/mg-air/yr)'
            else:
                gunit = '(Tg/yr)'
            get_budget_plot(df,path,col,ind,case1,case2,unit=gunit)
        pd.options.display.float_format = '{:g}'.format
        df = df.applymap(lambda x: rounding(x))
        df = df.astype(str)
        dfhtml = get_html_table(df)
        htable = dfhtml.replace('<thead>','\n<caption style = "font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 5px;width: auto"><strong>CNTL:</strong>   '+case1+'</caption>\n<caption style = "font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 5px;width: auto"><strong>TEST:</strong>   '+case2+'</caption>\n<caption style = "font-family: Century Gothic, sans-serif;font-size: medium;text-align: left;padding: 5px;width: auto"><strong>VRBL:</strong>  '+col+'</caption>\n<thead>')
        with open(path+'/'+col+'_'+ss[ind]+'.html','w') as f:
            f.write(htable)
    return cdatarel.columns

def gather_data(path,aer,case,mod,plev=None,sv=None,fact=1,vertinit=None,unit=None,reg=None,land=None):
    ss = ['ANN','DJF','JJA']
    dlist = []
    mlist = []
    for s in ss:
        if sv!=None:
            orig=get_singleV_hplots(path,case,s,aer,mod=mod,fact=1,vertinit=None,pval='radiation')
        else:
            orig=get_hplots(path,case,s,aer,mod=mod,plev=plev,reg=reg,land=land)
        dlist.append(orig[0])
        mlist.append(orig[1])
    data_combined=xr.concat(dlist,"season")
    m_combined=xr.concat(mlist,"season")
    if unit==None:
        if plev==None: 
            if aer=='num':
                unit = "[# $m^{-2}$]"
            else:
                unit = "[ug $m^{-2}$]"
        else:
            if aer=='num':
                unit = "[# $m^{-3}$]"
            else:
                unit = "[ug $m^{-3}$]"
    return data_combined,m_combined,orig[2],orig[3],unit,orig[4],orig[5]

def gather_ProfData(path,aer,case,mod,sv=None,lats=None,lons=None):
    ss = ['ANN','DJF','JJA']
    dlist = []
    plist = []
    for s in ss:
        if sv != None:
            orig=get_svplots(path,case,s,aer,mod=mod,lats=lats,lons=lons)
            dlist.append(orig[0])
            plist.append(orig[2])
        else:
            orig=get_vplots(path,case,s,aer,mod=mod,lats=lats,lons=lons)
            dlist.append(orig[0])
            plist.append(orig[2])
    data_combined=xr.concat(dlist,"season")
    pdata_combined=xr.concat(plist,"season")
        
    return data_combined,orig[1],pdata_combined

def get_latlon(reg):
    regions = {'CONUS':'24.74 49.34 -124.78 -66.95',\
              'NA':'15 72 -167 -50',\
              'EUS':'24.74 49.34 -97 -66.95',\
              'ECN':'18 45 90 130',\
              'IND':'6 40 66 98',\
              'CAF':'-5 20 -18 50', \
               'SH_pole':'-90 -60 -180 180',\
              'SH_midlat':'-60 -30 -180 180',\
              'Tropics':'-30 30 -180 180',\
              'NH_midlat':'30 60 -180 180',\
              'NH':'0 90 -180 180',\
              'SH':'-90 0 -180 180',\
              'NH_pole':'60 90 -180 180',\
              'Global':'-90 90 -180 180'}
    lat1 = float(regions[reg].split(' ')[0])
    lat2 = float(regions[reg].split(' ')[1])
    lon1 = float(regions[reg].split(' ')[2])
    lon2 = float(regions[reg].split(' ')[3])
    return lat1,lat2,lon1,lon2

def forcing_plots(plot_vars,path,area,season,plane,lon1,lon2,lat1,lat2,scrip):
    titles=['TOA $\u0394$F : ALL','TOA $\u0394$F$_{SW}$ : ALL','TOA $\u0394$F$_{LW}$ : ALL',\
               'TOA $\u0394$F : IND','TOA $\u0394$F$_{SW}$ : IND','TOA $\u0394$F$_{LW}$ : IND',\
               'TOA $\u0394$F : DIR','TOA $\u0394$F$_{SW}$ : DIR','TOA $\u0394$F$_{LW}$ : DIR',\
               'TOA $\u0394$F : clear-sky DIR','TOA $\u0394$F$_{SW}$ : clear-sky DIR','TOA $\u0394$F$_{LW}$ : clear-sky DIR',\
               'TOA $\u0394$F : RES','TOA $\u0394$F$_{SW}$ : RES','TOA $\u0394$F$_{LW}$ : RES']
    labels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)',\
           '(k)','(l)','(m)','(n)','(o)']
    plt.figure(figsize=(18,16))
    rr = [-20.,-10.,-5.,-2.,-1.,-0.5,0.5,1.,2.,5.,10.,20.]
    i=1
    for var,t,l in zip(plot_vars,titles,labels):
        ax=plt.subplot(5,3,0+i,projection=crs.PlateCarree())
        m=(var*area).sum(area.dims)/(area).sum(area.dims)
        try:
            get_plots( var,ax=ax,cmap=BlueWhiteOrangeRed_map,levels=rr,\
                         scrip_file=scrip,gridLines=False,\
                            lon_range=[lon1,lon2], lat_range=[lat1,lat2],
                            unit='[W m$^{-2}$]').get_map()
        except:
            get_plots( var,ax=ax,cmap=BlueWhiteOrangeRed_map,levels=rr,\
                         scrip_file='',gridLines=False,\
                            lon_range=[lon1,lon2], lat_range=[lat1,lat2],
                            unit='[W m$^{-2}$]').get_map()
        ax.text(0.005,1.03,t,size=12,transform=ax.transAxes)
        ax.text(0.89,1.03, '{:0.3f}'.format(m.values),size=12,transform=ax.transAxes)
        ax.text(0.05,0.95,l,size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
        i+=1
    plt.savefig(str(path)+'/'+plane+'_forcing_'+season+'_latlon.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
    
    
def get_forcings(datadef,datase,lon,lat,area,path,season,scrip,reg=None,loc=None):
    
    if reg!=None:
        lat1,lat2,lon1,lon2=get_latlon(reg)
    elif loc!=None:
        lat1,lon1=get_local(loc)
        lat1,lat2,lon1,lon2 = get_nearestlatlon(lon1,lat1,lon,lat)
    else:
        lat1,lat2,lon1,lon2=lat.values.min(),lat.values.max(),lon.values.min(),lon.values.max()

    SWIND = (datase['FSNT_d1'] - datase['FSNTC_d1']) - (datadef['FSNT_d1'] - datadef['FSNTC_d1'])
    SWDIR = (datase['FSNT'] - datase['FSNT_d1']) - (datadef['FSNT'] - datadef['FSNT_d1'])
    SWDIR_C = (datase['FSNTC'] - datase['FSNTC_d1']) - (datadef['FSNTC'] - datadef['FSNTC_d1'])
    SWALB = datase['FSNTC_d1'] - datadef['FSNTC_d1']
    LWIND = -1 * ( (datase['FLNT_d1'] - datase['FLNTC_d1']) - (datadef['FLNT_d1'] - datadef['FLNTC_d1']) )
    LWDIR = -1 * ( (datase['FLNT'] - datase['FLNT_d1']) - (datadef['FLNT'] - datadef['FLNT_d1']) )
    LWALB = -1 * ( datase['FLNTC_d1'] - datadef['FLNTC_d1'] )
    AIND = SWIND + LWIND
    ADIR = SWDIR + LWDIR
    AALB = SWALB + LWALB
    AA = AIND + ADIR + AALB
    TTAEF = -1*(datadef['FSNT'] - datadef['FLNT'] - (datase['FSNT'] - datase['FLNT']))
    SWAEF = -1*(datadef['FSNT'] - datase['FSNT'])
    LWAEF = (datadef['FLNT'] - datase['FLNT'])
    SWCAEF = -1*(datadef['FSNTC'] - datase['FSNTC'])
    SWCAEF_clean = -1*(datadef['FSNTC_d1'] - datase['FSNTC_d1'])
    LWCAEF = (datadef['FLNTC'] - datase['FLNTC'])
    LWCAEF_clean = 1*(datadef['FLNTC_d1'] - datase['FLNTC_d1'])
    TTIND = SWIND + LWIND
    TTDIR   = SWDIR + LWDIR
    SWCDIR = (SWCAEF - SWCAEF_clean)
    LWCDIR = (LWCAEF - LWCAEF_clean)
    TTCDIR   = SWCDIR + LWCDIR
    TTALB = SWALB + LWALB
    
    if reg=='Global':
        plot_vars=[TTAEF,SWAEF,LWAEF,TTIND, SWIND,LWIND,TTDIR,SWDIR,LWDIR,\
                  TTCDIR,SWCDIR,LWCDIR,TTALB,SWALB,LWALB]
        names=['TTAEF', 'SWAEF', 'LWAEF', 'TTIND', ' SWIND', 'LWIND', 'TTDIR', 'SWDIR', 'LWDIR', 'TTCDIR', 'SWCDIR', 'LWCDIR', 'TTALB', 'SWALB', 'LWALB']
        for n,v in zip(names,plot_vars):
            v.name = n.strip()
        saving_data = xr.merge(plot_vars)
        #saving_data.load().to_netcdf(path+'/TOA_forcing_vars_global.nc')
        forcing_plots(plot_vars,path,area,season,'TOA',lon1,lon2,lat1,lat2,scrip)

    all_vars=[TTAEF,SWAEF,LWAEF,SWCAEF,LWCAEF,SWIND,LWIND,TTIND,SWDIR,LWDIR,TTDIR,\
             SWCDIR,LWCDIR,TTCDIR,SWALB,LWALB,TTALB]
    all_means=[]
    for vdata in all_vars:
        vdatalatlon = vdata.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
        arealatlon = area.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
        mean = (vdatalatlon*arealatlon).sum(arealatlon.dims)/(arealatlon).sum(arealatlon.dims)
        all_means.append(mean.values)
    #########
    #########
    SWIND = (datase['FSNS_d1'] - datase['FSNSC_d1']) - (datadef['FSNS_d1'] - datadef['FSNSC_d1'])
    SWDIR = (datase['FSNS'] - datase['FSNS_d1']) - (datadef['FSNS'] - datadef['FSNS_d1'])
    SWDIR_C = (datase['FSNSC'] - datase['FSNSC_d1']) - (datadef['FSNSC'] - datadef['FSNSC_d1'])
    SWALB = datase['FSNSC_d1'] - datadef['FSNSC_d1']
    LWIND = -1 * ( (datase['FLNS_d1'] - datase['FLNSC_d1']) - (datadef['FLNS_d1'] - datadef['FLNSC_d1']) )
    LWDIR = -1 * ( (datase['FLNS'] - datase['FLNS_d1']) - (datadef['FLNS'] - datadef['FLNS_d1']) )
    LWALB = -1 * ( datase['FLNSC_d1'] - datadef['FLNSC_d1'] )
    AIND = SWIND + LWIND
    ADIR = SWDIR + LWDIR
    AALB = SWALB + LWALB
    AA = AIND + ADIR + AALB
    TTAEF = -1*(datadef['FSNS'] - datadef['FLNS'] - (datase['FSNS'] - datase['FLNS']))
    SWAEF = -1*(datadef['FSNS'] - datase['FSNS'])
    LWAEF = (datadef['FLNS'] - datase['FLNS'])
    SWCAEF = -1*(datadef['FSNSC'] - datase['FSNSC'])
    SWCAEF_clean = -1*(datadef['FSNSC_d1'] - datase['FSNSC_d1'])
    LWCAEF = (datadef['FLNSC'] - datase['FLNSC'])
    LWCAEF_clean = 1*(datadef['FLNSC_d1'] - datase['FLNSC_d1'])
    TTIND = SWIND + LWIND
    TTDIR   = SWDIR + LWDIR
    SWCDIR = (SWCAEF - SWCAEF_clean)
    LWCDIR = (LWCAEF - LWCAEF_clean)
    TTCDIR   = SWCDIR + LWCDIR
    TTALB = SWALB + LWALB

    if reg=='Global':
        plot_vars=[TTAEF,SWAEF,LWAEF,TTIND, SWIND,LWIND,TTDIR,SWDIR,LWDIR,\
                  TTCDIR,SWCDIR,LWCDIR,TTALB,SWALB,LWALB]
        names=['TTAEF', 'SWAEF', 'LWAEF', 'TTIND', ' SWIND', 'LWIND', 'TTDIR', 'SWDIR', 'LWDIR', 'TTCDIR', 'SWCDIR', 'LWCDIR', 'TTALB', 'SWALB', 'LWALB']
        for n,v in zip(names,plot_vars):
            v.name = n.strip()
        saving_data = xr.merge(plot_vars)
        #saving_data.load().to_netcdf(path+'/SFC_forcing_vars_global.nc')
        forcing_plots(plot_vars,path,area,season,'SFC',lon1,lon2,lat1,lat2,scrip)
    
    all_vars=[TTAEF,SWAEF,LWAEF,SWCAEF,LWCAEF,SWIND,LWIND,TTIND,SWDIR,LWDIR,TTDIR,\
             SWCDIR,LWCDIR,TTCDIR,SWALB,LWALB,TTALB]
    for vdata in all_vars:
        vdatalatlon = vdata.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
        arealatlon = area.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
        mean = (vdatalatlon*arealatlon).sum(arealatlon.dims)/(arealatlon).sum(arealatlon.dims)
        all_means.append(mean.values)   
    return all_means


def get_forcing_df(path1,path2,case1,case2,path,season='ANN',mod='eam',\
                   regions=['Global','SH_pole','SH_midlat','Tropics','NH_midlat','NH_pole'],\
                   scrip='/compyfs/www/hass877/share/emis_data/DECK120_to_SE/northamericax4v1pg2_scrip.nc'):
    
    try:
        datadef = xr.open_mfdataset(path1+case1+'.'+mod+'.'+season+'.*_climo.nc')
        lon = datadef['lon'].values
        lon[lon > 180.] -= 360.
    except:
        datadef = xr.open_mfdataset(path1+case1+'*'+season+'*_climo.nc').isel(time=0)
        lon = xr.where(datadef.lon > 180,datadef.lon-360,datadef.lon)
        lon = lon.assign_coords(lon=lon.values)
        datadef['lon'] = lon
        lon = lon.sortby(lon)
        datadef = datadef.sortby('lon')
    
    try:
        datase = xr.open_mfdataset(path2+case2+'.'+mod+'.'+season+'.*_climo.nc')
        lon = datase['lon'].values
        lon[lon > 180.] -= 360.
    except:
        datase = xr.open_mfdataset(path2+case2+'*'+season+'*_climo.nc').isel(time=0)
        lon = xr.where(datase.lon > 180,datase.lon-360,datase.lon)
        lon = lon.assign_coords(lon=lon.values)
        datase['lon'] = lon
        lon = lon.sortby(lon)
        datase = datase.sortby('lon')
        
    lat = datase['lat']
    
    varlist=['AODVIS','FSNT','FLNT','FSNTC','FLNTC','FSNT_d1','FLNT_d1',\
             'FSNTC_d1','FLNTC_d1','FSNS','FLNS','FSNSC','FLNSC','FSNS_d1',\
             'FLNS_d1','FSNSC_d1','FLNSC_d1']
    area = datase['area']
    datadef=datadef[varlist]
    datase=datase[varlist]
    lon[lon > 180.] -= 360.
    var_names=['TTAEF', 'SWAEF', 'LWAEF', 'SWCAEF', 'LWCAEF', 'SWIND', 'LWIND', \
               'TTIND', 'SWDIR', 'LWDIR', 'TTDIR', 'SWCDIR', 'LWCDIR', 'TTCDIR', \
               'SWALB', 'LWALB', 'TTALB','TTAEFs', 'SWAEFs', 'LWAEFs', 'SWCAEFs', \
               'LWCAEFs', 'SWINDs', 'LWINDs', 'TTINDs', 'SWDIRs', 'LWDIRs', \
               'TTDIRs', 'SWCDIRs', 'LWCDIRs', 'TTCDIRs', 'SWALBs', 'LWALBs', \
               'TTALBs'] 
    df = pd.DataFrame()
    for reg in regions:
        df[reg] = get_forcings(datadef,datase,lon,lat,area,path,season,scrip,reg=reg)
    df.index=var_names
    df.to_csv(path+'/AllForcings_'+season+'.csv',index=False)
    pd.options.display.float_format = '{:g}'.format
    df = df.applymap(lambda x: rounding(x))
    df = df.astype(str)
    htable = get_html_table(df)
    with open(path+'/'+'AllForcings_'+season+'.html','w') as f:
        f.write(htable)

def getVmap(data,ranges,ax,unit,cm=plt.cm.jet,cbs=0,cbi=1,cbe=-1):
    clen=len(np.arange(0,257)[cbs:cbe:cbi])
    try:
        cmap = ListedColormap(cm.colors[cbs:cbe:cbi])
    except:
        cmap = cm
        print('Cannot subscript Segmented Colormap!')
    x,y = np.meshgrid(data['lat'],data['lev'])
    im=ax.contourf(x,y,data[:],cmap=cmap,levels=ranges,norm=matplotlib.colors.BoundaryNorm(boundaries=ranges, ncolors=clen),extend='both')
    plt.gca().invert_yaxis()
    plt.xlim([-89,88])
    plt.xticks([-60,-30,0,30,60])
    ax.set_xticklabels(['60S','30S','0','30N','60N'],size=12)
    ax.yaxis.set_tick_params(width=1.5,length=5)
    ax.xaxis.set_tick_params(width=1.5,length=5)
    ax.grid( lw=0.5, color='#EBE7E0', alpha=0.5, linestyle='-.')
    ax.tick_params(labelsize=12)
    cbar = plt.colorbar(im,ticks=ranges,drawedges=True,extendrect=True)
    s1 = pd.DataFrame(ranges)
    s2 = s1.applymap(lambda x: rounding(x))[0].tolist()
    cbar_ticks=list(map(str,s2))
    cbar_ticks = [i.replace('.0','') if i[-2:]=='.0' else i for i in cbar_ticks]
    cbar.ax.set_yticklabels(cbar_ticks,size=12)
    cbar.set_label(label=unit,size=12)
    cbar.outline.set_linewidth(1.5)
    cbar.dividers.set_linewidth(1.5)
    plt.setp(ax.spines.values(),lw=1.5)
    plt.ylabel('Pressure [hPa]',fontsize=15)

def get_vert_profiles(data1,data2,diff,rel,var,ind,case1,case2,path=None,gunit=None):
    dd1=data1.isel(season=ind)
    dd2=data2.isel(season=ind)
    rr=gen_colbar_range(v1=dd1,v2=dd2).vmap()
    aagg=(np.max(dd1).values+np.max(dd2).values)/2
    print('var',var)
    print(aagg)
    ee=diff.isel(season=ind)
    rr_diff=gen_colbar_range(diff=ee,v1=dd1).vdiff()
    ff=rel.isel(season=ind)
    rr_rel=[-100.,-50.,-20.,-10.,-5.,-2.,2.,5.,10.,20.,50.,100.]

    ss = ['ANN','DJF','JJA']
    titles = ['Control Case','Test Case','Test Case'+' $-$ '+'Control Case','Relative diff (%)']
    colBars = [rr,rr,rr_diff,rr_rel]
    colMaps = [amwg256_map,amwg256_map,BlueWhiteOrangeRed_map,BlueWhiteOrangeRed_map]

    if gunit == None:
        if 'num' in var:
            gunit = '[# cm$^{-3}$]'
        else:
            gunit = '[ug m$^{-3}$]'

    units = [gunit,gunit,gunit,'[%]']
    varbls = [dd1,dd2,ee,ff]

    fig = plt.figure(figsize=(18,14))

    for i,t,colr,u,cmap,vals in zip([1,2,3,4],titles,colBars,units,colMaps,varbls):
        if i<3:
            if aagg == 0:
                cbs=-20
                cbe=5
                cbi=-2
            else:
                cbs=5
                cbe=-20
                cbi=2
        else:
            if aagg == 0:
                cbs=-1
                cbe=0
                cbi=-5
            else:
                cbs=0
                cbe=-1
                cbi=5
        panel=plt.subplot(220+i)
        getVmap(vals,colr,panel,u,cm=cmap,cbs=cbs,cbe=cbe,cbi=cbi)
        panel.text(0.005,1.03,t,size=15,transform=panel.transAxes)

    fig.suptitle(r'$\bf{Control\ Case:}$ '+case1+'\n'+\
                 r'$\bf{Test\ Case:}$ '+case2+'\n'+r'$\bf{Plotting:}$ '+var,\
                 fontsize=20,horizontalalignment='left',x=0.125,y=0.98)
    ## Saving figure
    plt.savefig(str(path)+'/'+var+'_'+ss[ind]+'_lathgt.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)

def get_local_profiles(data1,data2,diff,rel,var,case1,case2,loc,path=None,gunit=None):
    if gunit == None:
        if 'num' in var:
            gunit = '[# cm$^{-3}$]'
        else:
            gunit = '[ug m$^{-3}$]'
        units = [gunit,gunit,gunit,'[%]']
    
    varbls = [data1,data2,diff,rel]
    titles = ['Control Case','Test Case','Test Case'+' $-$ '+'Control Case','Relative diff (%)']

    fig = plt.figure(figsize=(14,14))

    for i,u,vals,t in zip([1,2,3,4],units,varbls,titles):
        ax=plt.subplot(220+i)
        plt.plot(vals.isel(season=0),vals['grid'],color=(0.8705882352941177, 0.5607843137254902, 0.0196078431372549),label='ANN')
        plt.plot(vals.isel(season=1),vals['grid'],color='gray',label='DJF')
        plt.plot(vals.isel(season=2),vals['grid'],color='gray',label='JJA',linestyle='--')
        ranges = list(plt.xticks()[0])
        s1 = pd.DataFrame(ranges)
        s2 = s1.applymap(lambda x: rounding(x))[0].tolist()
        cbar_ticks=list(map(str,s2))
        plt.xticks(ranges,cbar_ticks)
        plt.gca().invert_yaxis()
        ax.yaxis.set_tick_params(width=1.5,length=5)
        ax.xaxis.set_tick_params(width=1.5,length=5)
        ax.grid( lw=0.5, color='#EBE7E0', alpha=0.5, linestyle='-.')
        ax.tick_params(labelsize=12)
        plt.setp(ax.spines.values(),lw=1.5)
        plt.ylabel('Pressure [hPa]',fontsize=15)
        plt.xlabel(u,fontsize=15)
        ax.text(0.005,1.03,t,size=15,transform=ax.transAxes)
        if i==1:
            plt.legend(fontsize=12)
            
    fig.suptitle(r'$\bf{Control\ Case:}$ '+case1+'\n'+\
                 r'$\bf{Test\ Case:}$ '+case2+'\n'+r'$\bf{Plotting:}$ '+var,\
                 fontsize=20,horizontalalignment='left',x=0.125,y=0.98)
    ## Saving figure
    plt.savefig(str(path)+'/'+var+'_'+loc+'_lathgt.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
    
def get_map(data1,data2,diff,rel,var,ind,case1,case2,mean1,mean2,pval,unit,lon,lat,scrip=None,reg='Global',path=None,grid=True):
    if reg!=None:
        lat1,lat2,lon1,lon2=get_latlon(reg)
    else:
        lat1,lat2,lon1,lon2=lat.min(),lat.max(),lon.min(),lon.max()
    if path==None:
        path = Path('.').absolute()
    if (reg == 'Global') or (reg == None):
        grid = False 
    dd1=data1.isel(season=ind)
    var1 = dd1.where((lon>=lon1) & (lon<=lon2))
    var1 = var1.where((lat>=lat1) & (lat<=lat2))
    var1 = var1.stack(grid=var1.dims)
    var1 = var1.dropna("grid", how="all")
    dd2=data2.isel(season=ind)
    var2 = dd2.where((lon>=lon1) & (lon<=lon2))
    var2 = var2.where((lat>=lat1) & (lat<=lat2))
    var2 = var2.stack(grid=var2.dims)
    var2 = var2.dropna("grid", how="all")
    rr=gen_colbar_range(v1=var1,v2=var2).hmap()
    ee=diff.isel(season=ind)
    eevar = ee.where((lon>=lon1) & (lon<=lon2))
    eevar = eevar.where((lat>=lat1) & (lat<=lat2))
    eevar = eevar.stack(grid=eevar.dims)
    eevar = eevar.dropna("grid", how="all")
    ff=rel.isel(season=ind)
    rr_diff=gen_colbar_range(diff=eevar).hdiff()
    rr_rel=[-100,-70,-50,-20,-10,-5,-2,2,5,10,20,50,70,100]
    m1 = mean1.isel(season=ind).values
    m2 = mean2.isel(season=ind).values
    m3 = m2-m1
    m4 = (m3/abs(m1))*100
    ss = ['ANN','DJF','JJA']
    titles = ['Control Case','Test Case','Test Case'+' $-$ '+'Control Case','Relative diff (%)']
    means = [m1,m2,m3,m4]
    colBars = [rr,rr,rr_diff,rr_rel]
    colMaps = [amwg256_map,amwg256_map,BlueWhiteOrangeRed_map,BlueWhiteOrangeRed_map]
    units = [unit,unit,unit,'[%]']
    varbls = [dd1,dd2,ee,ff]

    fig = plt.figure(figsize=(18,12))

    for i,t,m,colr,u,cm,vals in zip([1,2,3,4],titles,means,colBars,units,colMaps,varbls):
        panel=plt.subplot(220+i,projection=crs.PlateCarree())
        if i<3:
            cbs=5
            cbe=-20
            cbi=2
        else:
            cbs=0
            cbe=-1
            cbi=5
        try:
            get_plots( vals,ax=panel,cmap=cm,levels=colr,\
                         scrip_file=scrip,figsize=fig,gridLines=grid,\
                            lon_range=[lon1,lon2], lat_range=[lat1,lat2],
                            unit=u,cbs=cbs,cbe=cbe,cbi=cbi).get_map()
        except:
            get_plots( vals,ax=panel,cmap=cm,levels=colr,\
                         scrip_file='',figsize=fig,gridLines=grid,\
                            lon_range=[lon1,lon2], lat_range=[lat1,lat2],
                            unit=u,cbs=cbs,cbe=cbe,cbi=cbi).get_map()
        panel.text(0.005,1.03,t,size=15,transform=panel.transAxes)
        panel.text(0.8,1.03, 'mean: '+str(rounding(m)),size=15,transform=panel.transAxes)
    
    fig.suptitle(r'$\bf{CNTL:}$ '+case1+'\n'+\
                 r'$\bf{TEST:}$ '+case2+'\n'+r'$\bf{VRBL:}$ '+var,\
                 fontsize=20,horizontalalignment='left',x=0.125,y=0.96)
    ## Saving figure
    plt.savefig(str(path)+'/'+var+'_'+ss[ind]+'_latlon_'+pval+'.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
    plt.close()



