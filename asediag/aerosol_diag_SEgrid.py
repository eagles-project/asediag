import xarray as xr
import numpy as np
## imports for plots
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cmaps
from asediag.Plot_2D import Plot_2D
import matplotlib
import fnmatch
from pathlib import Path
import pandas as pd
import math
from pretty_html_table import build_table

##########################################################################
##########################################################################3
def get_crange(v1,v2):
    aagg = (np.max(v1.values)+np.max(v2.values))/2
    aagg = np.log10(aagg)
    expo = np.floor(aagg)
    bbgg = aagg - expo
    if 10**(bbgg)<2.:
        s1 = [5*10**(expo-4),1*10**(expo-3),2*10**(expo-3), \
                5*10**(expo-3),1*10**(expo-2),2*10**(expo-2), \
                5*10**(expo-2),1*10**(expo-1),2*10**(expo-1), \
                5*10**(expo-1),10**expo,      2.*10**expo]
    elif 10**(bbgg)<5.:
        s1 = [1*10**(expo-3),2*10**(expo-3),5*10**(expo-3), \
                1*10**(expo-2),2*10**(expo-2),5*10**(expo-2), \
                1*10**(expo-1),2*10**(expo-1),5*10**(expo-1), \
                10**expo,      2.*10**expo,   5.*10**expo]
    else:
        s1 = [2*10**(expo-3),5*10**(expo-3),1*10**(expo-2), \
                2*10**(expo-2),5*10**(expo-2),1*10**(expo-1), \
                2*10**(expo-1),5*10**(expo-1),10**expo,       \
                2.*10**expo,   5.*10**expo,   10**(expo+1)]
    return s1

def get_crange2(diff):
    aagg = np.max(abs(diff).values)
    aagg = np.log10(aagg)
    expo = np.ceil(aagg)
    s2 = np.array([-100.,-50.,-20.,-10.,-5.,-2.,-1.,1.,2.,5.,10.,20.,50.,100.])*(10**(expo)/100)
    print(s2)
    return s2

def rounding(n):
    try:
        sgn = -1 if n<0 else 1
        num = format(abs(n)-int(abs(n)),'f')
        for i,e in enumerate(num[2:]):
            if e!= '0':
                if i==0:
                    d = int(abs(n)) + float(num[:i+5])
                else:
                    d = int(abs(n)) + float(num[:i+4])
                return sgn * d
    except:
        return np.nan

def get_vertint(vdata,ha,p0,hb,ps,grav,fact):
    ## calc. dp
    delp = 0*vdata
    p = ha*p0+hb*ps
    p = p.transpose('ilev','ncol')
    delp = p[1:,:].values-p[:-1,:].values
    delp = delp + 0*vdata
    ## unit conversion and vertical integration
    vdata = vdata*(delp/grav) # p/g = Pa/ms^-2 = Nm^-2/ms^-2 = Kg.ms^-2/m^2/ms^-2
    vdata = vdata*fact
    vdata = vdata.sum('lev')
    return vdata
    
def get_hplots(path,case,ts,aer,plev=None,mod='eam'):
    ## reading data as xarray
    data = xr.open_mfdataset(path+case+'.'+mod+'.'+ts+'.*_climo.nc')
    lon = data['lon']
    lon[lon > 180.] -= 360.
    lat = data['lat']
    if ts=='ANN':
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
    ## all variable list
    vlist = list(data.variables.keys())
    # Total BC burden
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
    ## getting total
    vdata[aer] = vdata.to_array().sum('variable')
    ## actual mean
    mean = (vdata*area).sum(['ncol'])/(area).sum(['ncol'])
    return vdata,mean,var_vars+[aer],pval,lon,lat

def get_singleV_hplots(path,case,ts,var,fact=1,vertinit=None,pval='radiation',mod='eam'):
    ## reading data as xarray
    data = xr.open_mfdataset(path+case+'.'+mod+'.'+ts+'.*_climo.nc')
    lon = data['lon']
    lon[lon > 180.] -= 360.
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
    mean = (vdata*area).sum(['ncol'])/(area).sum(['ncol'])
    return vdata,mean,[var],pval,lon,lat

def get_tables(path,case,ts,aer,reg=None,loc=None,mod='eam'):
    data = xr.open_mfdataset(path+case+'.'+mod+'.'+ts+'.*_climo.nc')
    lon = data['lon']
    lon[lon > 180.] -= 360.
    lat = data['lat']
    if reg!=None:
        lat1,lat2,lon1,lon2=get_latlon(reg)
    elif loc!=None:
        lat1,lon1=get_local(loc)
        lat1,lat2,lon1,lon2 = get_nearestlatlon(lon1,lat1,lon,lat)
    else:
        lat1,lat2,lon1,lon2=lat.values.min(),lat.values.max(),lon.values.min(),lon.values.max()
    if ts=='ANN':
        data = data.rename({'year':'season'})
    ## factors
    fact = 1e-9
    grav = 9.806
    ha = data['hyai']
    hb = data['hybi']
    p0 = data['P0']
    ps = data['PS']
    area = data['area']
    esfc=4*np.pi*(6.37122e6)**2
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
    psmean = (ps*area).sum(['ncol'])/(area).sum(['ncol'])
    sum_airmass = ((psmean*esfc)/grav)*1e6
    ## all variable list
    vlist = list(data.variables.keys())
    # List of all variables considered
    avariables = [aer+'_a?',aer+'_a?'+'DDF',aer+'_a?'+'SFWET','SF'+aer+'_a?',aer+'_a?'+'_CLXF',\
     aer+'_a?'+'_sfgaex1',aer+'_a?'+'GVF',aer+'_a?'+'TBF',aer+'_a?'+'SFSIS',\
     aer+'_a?'+'SFSIC',aer+'_a?'+'SFSBS',aer+'_a?'+'SFSBC',aer+'_a?'+'SFSES',\
     aer+'_a?'+'SFSEC',aer+'_a?'+'_sfgaex2',aer+'_a?'+'_sfcoag1',aer+'_a?'+'_sfcsiz3',\
     aer+'_a?'+'_sfcsiz4',aer+'_a?'+'_mixnuc1',aer+'_a?'+'AQH2SO4',\
     aer+'_a?'+'AQSO4',aer+'_a?'+'_sfnnuc1',aer+'_a?']
    cvariables = [aer+'_c?',aer+'_c?'+'DDF',aer+'_c?'+'SFWET','SF'+aer+'_c?',aer+'_c?'+'_CLXF',\
     aer+'_c?'+'_sfgaex1',aer+'_c?'+'GVF',aer+'_c?'+'TBF',aer+'_c?'+'SFSIS',\
     aer+'_c?'+'SFSIC',aer+'_c?'+'SFSBS',aer+'_c?'+'SFSBC',aer+'_c?'+'SFSES',\
     aer+'_c?'+'SFSEC',aer+'_c?'+'_sfgaex2',aer+'_c?'+'_sfcoag1',aer+'_c?'+'_sfcsiz3',\
     aer+'_c?'+'_sfcsiz4',aer+'_c?'+'_mixnuc1',aer+'_c?'+'AQH2SO4',\
     aer+'_c?'+'AQSO4',aer+'_c?'+'_sfnnuc1',aer+'_c?']
    # sfc emis
    df = pd.DataFrame()
    nvar=0
    for avar,cvar in zip(avariables[:],cvariables[:]):
        nvar+=1
        var_avars = fnmatch.filter(vlist,avar)
        var_cvars = fnmatch.filter(vlist,cvar)
        var_vars = var_avars+var_cvars
        #print(var_vars)
        vdata = data[var_vars]
        if (avar == aer+'_a?') and (nvar == 1):
            vars1 = var_vars+[avar+'+'+cvar]
            if aer == 'so4':
                bname = 'Burden (TgS)'
                srcname = 'Sources (TgS/yr)'
                snkname = 'Sinks (TgS/yr)'
                vdata = get_vertint(vdata,ha,p0,hb,ps,grav,factdd)*esfc
            elif aer == 'num':
                bname = 'Burden (#/mg-air)'
                srcname = 'Sources (#/mg-air/yr)'
                snkname = 'Sinks (#/mg-air/yr)'
                vdata = get_vertint(vdata,ha,p0,hb,ps,grav,1)*esfc/sum_airmass
            else:
                bname = 'Burden (Tg)'
                srcname = 'Sources (Tg/yr)'
                snkname = 'Sinks (Tg/yr)'
                vdata = get_vertint(vdata,ha,p0,hb,ps,grav,fact)*esfc
        elif (avar == aer+'_a?') and (nvar > 1):
            if aer == 'so4':
                #sfc conc
                sname = 'Sfc Conc. (ug/m3)'
                vdata = vdata[dict(lev=-1)].drop_vars('lev')
                vdata = vdata*1e9
            elif aer == 'num':
                #sfc conc
                sname = 'Sfc Conc. (#/m3)'
                vdata = vdata[dict(lev=-1)].drop_vars('lev')
                vdata = vdata*1
            else:
                #sfc conc
                sname = 'Sfc Conc. (ug/m3)'
                vdata = vdata[dict(lev=-1)].drop_vars('lev')
                vdata = vdata*1e9
        else:
            if ('_CLXF' in avar):
                if (aer == 'bc') or (aer == 'pom'):
                    vdata = vdata*factaaa*esfc*factbb
                elif aer == 'num':
                    vdata = vdata*factaa*esfc*factbb/sum_airmass
                elif aer == 'so4':
                    vdata = vdata*factaa*factcc*esfc
                else:
                    vdata = vdata*factaa*esfc
            else:
                if aer == 'so4':
                    vdata = vdata*factcc*esfc
                elif aer == 'num':
                    vdata = vdata*factbb*esfc/sum_airmass
                else:
                    vdata = vdata*factbb*esfc
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
        ## actual mean
        vdatalatlon = vdata.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
        arealatlon = area.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
        mean = (vdatalatlon*arealatlon).sum(['ncol'])/(arealatlon).sum(['ncol'])
        rvars = dict(zip(prob_list+[avar+'+'+cvar],vars1))
        mean = mean.rename_vars(rvars)
        if (('DDF' in avar) or ('GVF' in avar) or ('TBF' in avar)):
            mean = -1*mean
        ndf=mean.expand_dims(dim='vars').to_dataframe()
        df=pd.concat([df,ndf.replace(0, np.nan)])
    index_list = [bname,'Dry deposition','Wet deposition','surface emission',\
             'elevated emission','condensation-aging','gravitational','turbulent',\
             'incloud, stratiform','incloud, convective','belowcloud, strat.',\
              'belowcloud, convec.','rain evap, strat.','rain evap, convec.',\
             'renaming (sfgaex2)','coagulation (sfcoag1)','calcsize (sfcsiz3)',\
             'calcsize (sfcsiz4)','dropmixnuc (mixnuc1)','cloudchem (AQH2SO4)',\
             'cloudchem (AQSO4)','sfnnuc1',sname]
    df.index=index_list
    df.columns=df.columns.tolist()[:-1]+[aer]
    srcsnk = df.loc[['Dry deposition','Wet deposition','renaming (sfgaex2)',\
                     'coagulation (sfcoag1)','calcsize (sfcsiz3)',\
                     'calcsize (sfcsiz4)','dropmixnuc (mixnuc1)',\
                     'condensation-aging','surface emission','elevated emission',\
                     'cloudchem (AQH2SO4)','cloudchem (AQSO4)','sfnnuc1']][vars1[:-1]+[aer]]
    src = srcsnk.where(srcsnk>0).sum()
    snk = srcsnk.where(srcsnk<0).sum()
    df.loc[srcname] = src
    df.loc[snkname] = snk
    lifetime = (df.loc[bname][vars1[:-1]+[aer]]/abs(df.loc[snkname][vars1[:-1]+[aer]]))*365
    df.loc['Lifetime (days)'] = lifetime
    df['season']=ts
    df = df.reindex([bname,sname,srcname,'surface emission','elevated emission',snkname,\
               'Dry deposition','gravitational','turbulent','Wet deposition',\
               'incloud, stratiform','incloud, convective','belowcloud, strat.',\
               'belowcloud, convec.','rain evap, strat.','rain evap, convec.',\
               'Lifetime (days)','renaming (sfgaex2)','coagulation (sfcoag1)','calcsize (sfcsiz3)',\
                'calcsize (sfcsiz4)','dropmixnuc (mixnuc1)','cloudchem (AQH2SO4)',\
                'cloudchem (AQSO4)','condensation-aging'])
    return df

def get_all_tables(ind,aer,path1,path2,case1,case2,path,reg,loc,mod):
    ss = ['ANN','DJF','JJA']
    cdatadef=get_tables(path1,case1,ss[ind],aer,reg=reg,loc=loc,mod=mod)
    cdatase=get_tables(path2,case2,ss[ind],aer,reg=reg,loc=loc,mod=mod)
    cdatadiff = cdatase[cdatase.columns[1:]] - cdatadef[cdatase.columns[1:]]
    cdatarel = (cdatadiff/abs(cdatase[cdatase.columns[1:]]))*100
    for col in cdatarel.columns:
        df = pd.DataFrame()
        df[case1]=cdatadef[col]
        df[case2]=cdatase[col]
        df['difference']=cdatadiff[col]
        df['rel diff (%)']=cdatarel[col]
        pd.options.display.float_format = '{:g}'.format
        df = df.applymap(lambda x: rounding(x) if ((abs(x)>1e-5) and (abs(x)<1e5)) else '{:.0e}'.format(x))
        htable = build_table(df,'grey_light',index=True,padding='5px',text_align='right')
        with open(path+'/'+col+'_'+ss[ind]+'.html','w') as f:
            f.write(htable)

def gather_data(path,aer,case,plev=None,sv=None,fact=1,vertinit=None,unit=None):
    ss = ['ANN','DJF','JJA']
    dlist = []
    mlist = []
    for s in ss:
        if sv!=None:
            orig=get_singleV_hplots(path,case,s,aer,fact=1,vertinit=None,pval='radiation')
        else:
            orig=get_hplots(path,case,s,aer,plev=plev)
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
              'NH_pole':'60 90 -180 180',\
              'Global':'-90 90 -180 180'}
    lat1 = float(regions[reg].split(' ')[0])
    lat2 = float(regions[reg].split(' ')[1])
    lon1 = float(regions[reg].split(' ')[2])
    lon2 = float(regions[reg].split(' ')[3])
    return lat1,lat2,lon1,lon2

def get_local(reg):
    loclatlon = {'SGP':'36.605 -97.485',\
               'ENA':'39.091 -28.026',\
               'NSA':'71.322 -156.615',\
               'TCAP':'42.5 -70',\
               'TWP':'-2.06 147.425'}
    lat1 = float(loclatlon[reg].split(' ')[0])
    lon1 = float(loclatlon[reg].split(' ')[1])
    return lat1,lon1

def get_nearestlatlon(lon1,lat1,lon,lat):
    lon=lon.values
    lat=lat.values
    ind=np.argmin((lon-lon1)**2+(lat-lat1)**2)
    return lat[ind],lat[ind],lon[ind],lon[ind]

def forcing_plots(plot_vars,path,area,season,plane,lon1,lon2,lat1,lat2,scrip):
    titles=['TOA $\u0394$F : ALL','TOA $\u0394$F$_{SW}$ : ALL','TOA $\u0394$F$_{LW}$ : ALL',\
               'TOA $\u0394$F : IND','TOA $\u0394$F$_{SW}$ : IND','TOA $\u0394$F$_{LW}$ : IND',\
               'TOA $\u0394$F : DIR','TOA $\u0394$F$_{SW}$ : DIR','TOA $\u0394$F$_{LW}$ : DIR',\
               'TOA $\u0394$F : clear-sky DIR','TOA $\u0394$F$_{SW}$ : clear-sky DIR','TOA $\u0394$F$_{LW}$ : clear-sky DIR',\
               'TOA $\u0394$F : RES','TOA $\u0394$F$_{SW}$ : RES','TOA $\u0394$F$_{LW}$ : RES']
    labels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)',\
           '(k)','(l)','(m)','(n)','(o)']
    plt.figure(figsize=(18,22))
    rr = [-20,-10,-5,-2,-1,-.5,.5,1,2,5,10,20]
    i=1
    for var,t,l in zip(plot_vars,titles,labels):
        ax=plt.subplot(5,3,0+i,projection=crs.PlateCarree())
        m=(var*area).sum(['ncol'])/(area).sum(['ncol'])
        Plot_2D( var,ax=ax,cmap=cmaps.BlueWhiteOrangeRed,ranges=rr,\
             scrip_file=scrip,lon_range=[lon1,lon2], lat_range=[lat1,lat2],title=t,title_size=15,
                unit_offset=[-0.6,-2.2],pad=0.15,shrink=1,unit='[W m$^{-2}$]',unit_size=10,\
             country=False,extend='neither',resolution='50m',label_size=12).plot(rr)
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
        names=['TTAEF', 'SWAEF', 'LWAEF', 'TTIND', ' SWIND', 'LWIND', 'TTDIR', \
               'SWDIR', 'LWDIR', 'TTCDIR', 'SWCDIR', 'LWCDIR', 'TTALB', 'SWALB', 'LWALB']
        for n,v in zip(names,plot_vars):
            v.name = str(n.strip())
        saving_data = xr.merge(plot_vars)
        saving_data.load().to_netcdf(path+'/'+'TOA_forcing_vars_global.nc')
        forcing_plots(plot_vars,path,area,season,'TOA',lon1,lon2,lat1,lat2,scrip)

    all_vars=[TTAEF,SWAEF,LWAEF,SWCAEF,LWCAEF,SWIND,LWIND,TTIND,SWDIR,LWDIR,TTDIR,\
             SWCDIR,LWCDIR,TTCDIR,SWALB,LWALB,TTALB]
    all_means=[]
    for vdata in all_vars:
        vdatalatlon = vdata.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
        arealatlon = area.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
        mean = (vdatalatlon*arealatlon).sum(['ncol'])/(arealatlon).sum(['ncol'])
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
        names=['TTAEF', 'SWAEF', 'LWAEF', 'TTIND', ' SWIND', 'LWIND', 'TTDIR', \
               'SWDIR', 'LWDIR', 'TTCDIR', 'SWCDIR', 'LWCDIR', 'TTALB', 'SWALB', 'LWALB']
        for n,v in zip(names,plot_vars):
            v.name = str(n.strip())
        saving_data = xr.merge(plot_vars)
        saving_data.load().to_netcdf(path+'/'+'SFC_forcing_vars_global.nc')
        forcing_plots(plot_vars,path,area,season,'SFC',lon1,lon2,lat1,lat2,scrip)
    
    all_vars=[TTAEF,SWAEF,LWAEF,SWCAEF,LWCAEF,SWIND,LWIND,TTIND,SWDIR,LWDIR,TTDIR,\
             SWCDIR,LWCDIR,TTCDIR,SWALB,LWALB,TTALB]
    for vdata in all_vars:
        vdatalatlon = vdata.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
        arealatlon = area.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2))
        mean = (vdatalatlon*arealatlon).sum(['ncol'])/(arealatlon).sum(['ncol'])
        all_means.append(mean.values)   
    return all_means


def get_forcing_df(path1,path2,case1,case2,path,season='ANN',mod='eam',\
                   regions=['Global','SH_pole','SH_midlat','Tropics','NH_midlat','NH_pole'],\
                   scrip='/compyfs/www/hass877/share/emis_data/DECK120_to_SE/northamericax4v1pg2_scrip.nc'):
    datadef = xr.open_mfdataset(path1+case1+'.'+mod+'.'+season+'.*_climo.nc')
    datase = xr.open_mfdataset(path2+case2+'.'+mod+'.'+season+'.*_climo.nc')
    lon = datase['lon']
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
    df.to_csv(path+'/'+'AllForcings_'+season+'.csv',index=False)
    pd.options.display.float_format = '{:g}'.format
    df = df.applymap(lambda x: rounding(x) if ((abs(x)>1e-5) and (abs(x)<1e5)) else '{:.0e}'.format(x))
    htable = build_table(df,'grey_light',index=True,padding='5px',text_align='right')
    with open(path+'/'+'AllForcings_'+season+'.html','w') as f:
        f.write(htable)

def get_map(data1,data2,diff,rel,var,ind,case1,case2,mean1,mean2,pval,unit,lon,lat,reg=None,path=None):
    from Plot_2D import Plot_2D
    if reg!=None:
        lat1,lat2,lon1,lon2=get_latlon(reg)
    else:
        lat1,lat2,lon1,lon2=lat.min(),lat.max(),lon.min(),lon.max()
    if path==None:
        path = Path('.').absolute()
    ss = ['ANN','DJF','JJA']
    plt.figure(figsize=(18,12))
    ax=plt.subplot(221,projection=crs.PlateCarree())
    dd1=data1.isel(season=ind)
    var1 = dd1.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2)).dropna(dim='ncol')
    dd2=data2.isel(season=ind)
    var2 = dd2.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2)).dropna(dim='ncol')
    rr=get_crange(var1,var2)
    Plot_2D( dd1,ax=ax,cmap=cmaps.amwg256,ranges=rr,\
                 scrip_file='/compyfs/www/hass877/share/emis_data/DECK120_to_SE/northamericax4v1pg2_scrip.nc',\
                    lon_range=[lon1,lon2], lat_range=[lat1,lat2],
                    unit_offset=[-0.6,-2.2],pad=0.07,shrink=1,unit=unit,unit_size=12,\
                 grid_line=True, grid_line_lw=0.1).plot(rr)
    ax.text(0.005,1.03,case1,size=15,transform=ax.transAxes)
    ax.text(0.83,1.03, 'mean: '+'{:0.2e}'.format((mean1.isel(season=ind).values)),size=15,transform=ax.transAxes)
    ax=plt.subplot(222,projection=crs.PlateCarree())
    Plot_2D( dd2,ax=ax,cmap=cmaps.amwg256,ranges=rr,\
                 scrip_file='/compyfs/www/hass877/share/emis_data/DECK120_to_SE/northamericax4v1pg2_scrip.nc',\
                    lon_range=[lon1,lon2], lat_range=[lat1,lat2],
                    unit_offset=[-0.6,-2.2],pad=0.07,shrink=1,unit=unit,unit_size=12,\
                 grid_line=True, grid_line_lw=0.1).plot(rr)
    ax.text(0.005,1.03,case2,size=15,transform=ax.transAxes)
    ax.text(0.83,1.03, 'mean: '+'{:0.2e}'.format((mean2.isel(season=ind).values)),size=15,transform=ax.transAxes)
    ee=diff.isel(season=ind)
    eevar = ee.where((lon>=lon1) & (lon<=lon2) & (lat>=lat1) & (lat<=lat2)).dropna(dim='ncol')
    rr_diff=get_crange2(eevar)
    ax=plt.subplot(223,projection=crs.PlateCarree())
    Plot_2D( ee,ax=ax,cmap=cmaps.BlueWhiteOrangeRed,ranges=rr_diff,unit=unit,unit_size=12,\
                  scrip_file='/compyfs/www/hass877/share/emis_data/DECK120_to_SE/northamericax4v1pg2_scrip.nc',\
                    lon_range=[lon1,lon2], lat_range=[lat1,lat2],
                    unit_offset=[-0.6,-2.2],pad=0.07,shrink=1,\
                  grid_line=True, grid_line_lw=0.1).plot(rr_diff)
    ax.text(0.005,1.03,case2+' $-$ '+case1,size=15,transform=ax.transAxes)
    ax.text(0.83,1.03, 'mean: '+'{:0.2e}'.format((mean2.isel(season=ind).values-mean1.isel(season=ind).values)),size=15,transform=ax.transAxes)
    ff=rel.isel(season=ind)
    rr_rel=[-100,-50,-20,-10,-5,-2,2,5,10,20,50,100]
    ax=plt.subplot(224,projection=crs.PlateCarree())
    Plot_2D( ff,ax=ax,cmap=cmaps.BlueWhiteOrangeRed,ranges=rr_rel,unit='[%]',unit_size=12,\
                  scrip_file='/compyfs/www/hass877/share/emis_data/DECK120_to_SE/northamericax4v1pg2_scrip.nc',\
                    lon_range=[lon1,lon2], lat_range=[lat1,lat2],
                    unit_offset=[-0.6,-2.2],pad=0.07,shrink=1,\
                  grid_line=True, grid_line_lw=0.1).plot(rr_rel)
    ax.text(0.005,1.03,'Relative diff (%)',size=15,transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(str(path)+'/'+var+'_'+ss[ind]+'_latlon_'+pval+'.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)




