import fnmatch
import matplotlib.pyplot as plt
import cartopy.crs as crs
import xarray as xr
import numpy as np
from pathlib import Path

from asediag.asediag_utils import rounding, get_latlon, gen_colbar_range, get_vertint
from asediag.nclCols import amwg256_map, BlueWhiteOrangeRed_map
from asediag.aerdiag_plots import get_plots


class GenSpatialData:
    def __init__(self, path, case, aer, **kwargs):
        self.path = path
        self.case = case
        self.aer = aer
        self.mod = kwargs.get('mod', 'eam')
        self.pval = kwargs.get('pval', 'bdn')
        self.plev = kwargs.get('plev', None)
        # Optional parameters
        self.reg = kwargs.get('reg', None)
        self.land = kwargs.get('land', False)
        self.sv = kwargs.get('sv', None)
        self.unit = kwargs.get('unit', None)
        self.grid = kwargs.get('grid', True)
        self.scrip = kwargs.get('scrip', None)
        self.gvars = kwargs.get('gvars', ['SO2', 'DMS', 'H2SO4', 'SOAG'])
        self.seasons = kwargs.get('seasons', ['ANN', 'DJF', 'JJA'])

    def _open_dataset(self, ts):
        try:
            print('SE data:',f"{self.path}{self.case}.{self.mod}.{ts}.*_climo.nc")
            file_path = f"{self.path}{self.case}.{self.mod}.{ts}.*_climo.nc"
            data = xr.open_mfdataset(file_path)
            lon = data['lon']
            lon.load()
            lon[lon > 180.] -= 360.
        except:
            print('Lat/Lon data:',f"{self.path}{self.case}*{ts}_*_climo.nc")
            file_path = f"{self.path}{self.case}*{ts}_*_climo.nc"
            data = xr.open_mfdataset(file_path)
            if 'time' in data.dims:
                data = data.isel(time=0)
            lon = xr.where(data.lon > 180, data.lon - 360, data.lon)
            lon.load()
            lon = lon.assign_coords(lon=lon)
            data['lon'] = lon
            lon = lon.sortby(lon)
            data = data.sortby('lon')
        
        lat = data['lat']
        
        if 'year' in data.coords:
            data = data.rename({'year':'season'})
        data['season'] = ts
            
        return data, lon, lat

    def set_region(self):
        """Set the region for analysis."""
        if self.reg is not None:
            self.lat1, self.lat2, self.lon1, self.lon2 = get_latlon(self.reg)
        else:
            self.lat1, self.lat2, self.lon1, self.lon2 = self.lat.values.min(), self.lat.values.max(), self.lon.min(), self.lon.max()

    
    def calculate_factors(self):
        """Calculate and return conversion factors."""
        
        # Basic factors
        factors = {
            "fact": 1e9,
            "factaa": 1.01325e5 / 8.31446261815324 / 273.15 * 28.9647 / 1.e9,   # kg-air/cm3-air
            "factbb": None,
            "grav": 9.806,
        }
        
        # Calculation of factbb based on factaa
        factors["factbb"] = factors["factaa"] * 1.e15  # ug-air/m3-air
        model_levels = self.data.lev
        if self.plev:
            # Conditions based on pressure level (plev)
            tlev = model_levels[np.abs(model_levels - int(self.plev)).argmin()].values
            self.data = self.data.sel(lev=tlev)
            factors["fact"] = factors["factbb"]
            self.pval = self.plev

        if self.aer == 'num':
            if self.plev is not None:
                factors["fact"] = factors["factaa"]
            else:
                factors["fact"] = 1.0
        
        return factors

    def get_hplots_data(self, ts):
        
        self.data, self.lon, self.lat = self._open_dataset(ts)
        self.factors = self.calculate_factors()
        self.ps = self.data['PS']
        self.ha = self.data['hyai']
        self.hb = self.data['hybi']
        self.p0 = self.data['P0']
        self.area = self.data['area']
        self.landF = self.data['LANDFRAC']
        self.grav = self.factors["grav"]

        self.set_region()
        
        ## all variable list
        vlist = list(self.data.variables.keys())

        if self.aer in self.gvars:
            var_avars = fnmatch.filter(vlist,self.aer)
            var_cvars = []
        else:
            var_avars = fnmatch.filter(vlist,self.aer+'_a?')
            var_cvars = fnmatch.filter(vlist,self.aer+'_c?')
        var_vars = var_avars+var_cvars
        
        print(var_vars)
        vdata = self.data[var_vars]
        if self.plev == None:
            vdata = get_vertint(vdata,self.ha,self.p0,self.hb,self.ps,self.grav,self.factors["fact"])
        else:
            vdata = vdata*self.factors["fact"]
        if self.land==True:
            vdata = vdata.where(self.landF>0)
        else:
            vdata = vdata.where(self.landF>=0)
        
        ## getting total
        vdata[self.aer] = vdata.to_array().sum('variable')
        ## actual area weighted sums
        vdatalatlon = vdata.where((self.lon>=self.lon1) & (self.lon<=self.lon2) & (self.lat>=self.lat1) & (self.lat<=self.lat2))
        arealatlon = self.area.where((self.lon>=self.lon1) & (self.lon<=self.lon2) & (self.lat>=self.lat1) & (self.lat<=self.lat2))
        mean = (vdatalatlon*arealatlon).sum(vdatalatlon.dims)/(arealatlon).sum(arealatlon.dims)

        return vdata, mean, var_vars + [self.aer]

    def get_singleV_hplots(self, ts):
        
        self.data, self.lon, self.lat = self._open_dataset(ts)
        self.factors = self.calculate_factors()
        self.ps = self.data['PS']
        self.ha = self.data['hyai']
        self.hb = self.data['hybi']
        self.p0 = self.data['P0']
        self.area = self.data['area']
        self.landF = self.data['LANDFRAC']
        self.grav = self.factors["grav"]
        
        self.set_region()
        
        self.factors["fact"] = 1.0
        self.pval = 'radiation'
        vdata = self.data[self.aer]

        if ('ncol' in self.data.dims) and (len(vdata.dims) > 1):
            vdata = get_vertint(vdata,self.ha,self.p0,self.hb,self.ps,self.grav,self.factors["fact"])
        else:
            vdata = vdata * self.factors["fact"]

        ## actual area weighted sums
        vdatalatlon = vdata.where((self.lon>=self.lon1) & (self.lon<=self.lon2) & (self.lat>=self.lat1) & (self.lat<=self.lat2))
        arealatlon = self.area.where((self.lon>=self.lon1) & (self.lon<=self.lon2) & (self.lat>=self.lat1) & (self.lat<=self.lat2))
        mean = (vdatalatlon*arealatlon).sum(vdatalatlon.dims)/(arealatlon).sum(arealatlon.dims)
        
        return vdata, mean, self.aer

    def gather_data(self):
        dlist = []
        mlist = []
        for s in self.seasons:
            if self.sv:
                vdata, mean, var_vars = self.get_singleV_hplots(s)
            else:
                vdata, mean, var_vars = self.get_hplots_data(s)
            dlist.append(vdata)
            mlist.append(mean)

        data_combined = xr.concat(dlist, "season")
        m_combined = xr.concat(mlist, "season")

        if self.unit is None:
            if self.plev is None:
                self.unit = "[# $m^{-2}$]" if self.aer == 'num' else "[ug $m^{-2}$]"
            else:
                self.unit = "[# $m^{-3}$]" if self.aer == 'num' else "[ug $m^{-3}$]"

        return data_combined, m_combined, var_vars, self.pval, self.unit, self.lon, self.lat
 
def gen_4panel_maps(data1,data2,diff,rel,var,ind,case1,case2,mean1,mean2,pval,unit,lon,lat,scrip=None,reg='Global',path=None,grid=True):
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
