import fnmatch
import matplotlib.pyplot as plt
import cartopy.crs as crs
import xarray as xr
import numpy as np
from pathlib import Path

from src.utils.asediag_utils import rounding, get_latlon, gen_colbar_range, get_vertint
from src.utils.nclCols import amwg256_map, BlueWhiteOrangeRed_map
from src.utils.aerdiag_plots import get_plots


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
        ## actual area weighted means
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

        vlist = list(self.data.variables.keys())
        vars_list = [item for item in self.aer if item in vlist]
        NAvars = set(self.aer) - set(vars_list)
        if NAvars:
            print('\nThe following variables with vertical dimension are unavailable:\n',NAvars,'\n')
        vdata = self.data[vars_list]

        if ('ncol' in self.data.dims) and (len(vdata.dims) > 1):
            vdata = get_vertint(vdata,self.ha,self.p0,self.hb,self.ps,self.grav,self.factors["fact"])
        else:
            vdata = vdata * self.factors["fact"]

        ## actual area weighted means
        vdatalatlon = vdata.where((self.lon>=self.lon1) & (self.lon<=self.lon2) & (self.lat>=self.lat1) & (self.lat<=self.lat2))
        arealatlon = self.area.where((self.lon>=self.lon1) & (self.lon<=self.lon2) & (self.lat>=self.lat1) & (self.lat<=self.lat2))
        mean = (vdatalatlon*arealatlon).sum(vdatalatlon.dims)/(arealatlon).sum(arealatlon.dims)
        
        return vdata, mean, vars_list

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


class SpatialMapGenerator:
    def __init__(self, data1, data2, diff, rel, var, case1, case2, mean1, mean2, pval, unit, lon, lat, scrip=None, reg='Global', path=None, grid=True):
        self.data1 = data1
        self.data2 = data2
        self.diff = diff
        self.rel = rel
        self.var = var
        self.case1 = case1
        self.case2 = case2
        self.mean1 = mean1
        self.mean2 = mean2
        self.pval = pval
        self.unit = unit
        self.lon = lon
        self.lat = lat
        self.scrip = scrip
        self.reg = reg
        self.path = path if path is not None else Path('.').absolute()
        self.grid = grid
        self.lat1, self.lat2, self.lon1, self.lon2 = self.get_region_bounds()

    def get_region_bounds(self):
        """Get latitude and longitude bounds for the region."""
        if self.reg is not None:
            return get_latlon(self.reg)
        else:
            return self.lat.min(), self.lat.max(), self.lon.min(), self.lon.max()

    def process_data(self, data, ind):
        """Process data by selecting region and stacking grid."""
        dd = data.isel(season=ind)
        var = dd.where((self.lon >= self.lon1) & (self.lon <= self.lon2))
        var = var.where((self.lat >= self.lat1) & (self.lat <= self.lat2))
        var = var.stack(grid=var.dims)
        return dd, var.dropna("grid", how="all")

    def generate_colbar_ranges(self, var1, var2):
        """Generate color bar ranges for the given data."""
        return gen_colbar_range(v1=var1, v2=var2).hmap()

    def generate_4panel_maps(self, ind):
        """Generate 4-panel maps."""
        dd1, var1 = self.process_data(self.data1, ind)
        dd2, var2 = self.process_data(self.data2, ind)
        colbar_range = self.generate_colbar_ranges(var1, var2)

        ee, eevar = self.process_data(self.diff, ind)
        ff=self.rel.isel(season=ind)

        rel_diff_colbar = gen_colbar_range(diff=eevar).hdiff()
        rel_colbar = [-100, -70, -50, -20, -10, -5, -2, 2, 5, 10, 20, 50, 70, 100]

        m1 = self.mean1.isel(season=ind).values
        m2 = self.mean2.isel(season=ind).values
        m3 = m2 - m1
        m4 = (m3 / abs(m1)) * 100

        seasons = ['ANN', 'DJF', 'JJA']
        titles = ['Control Case', 'Test Case', 'Test Case $-$ Control Case', 'Relative diff (%)']
        means = [m1, m2, m3, m4]
        colBars = [colbar_range, colbar_range, rel_diff_colbar, rel_colbar]
        colMaps = [amwg256_map, amwg256_map, BlueWhiteOrangeRed_map, BlueWhiteOrangeRed_map]
        units = [self.unit, self.unit, self.unit, '[%]']
        varbls = [dd1, dd2, ee, ff]

        fig = plt.figure(figsize=(18, 12))

        for i, (title, mean, colr, u, cm, vals) in enumerate(zip(titles, means, colBars, units, colMaps, varbls), start=1):
            panel = plt.subplot(220 + i, projection=crs.PlateCarree())
            cbs, cbe, cbi = (5, -20, 2) if i < 3 else (0, -1, 5)

            try:
                get_plots( vals,ax=panel,cmap=cm,levels=colr,\
                            scrip_file=self.scrip,figsize=fig,gridLines=self.grid,\
                            lon_range=[self.lon1,self.lon2], lat_range=[self.lat1,self.lat2],
                            unit=u,cbs=cbs,cbe=cbe,cbi=cbi).get_map()
            except:
                get_plots( vals,ax=panel,cmap=cm,levels=colr,\
                            scrip_file='',figsize=fig,gridLines=self.grid,\
                            lon_range=[self.lon1,self.lon2], lat_range=[self.lat1,self.lat2],
                            unit=u,cbs=cbs,cbe=cbe,cbi=cbi).get_map()
            panel.text(0.005, 1.03, title, size=15, transform=panel.transAxes)
            panel.text(0.8,1.03, 'mean: '+str(rounding(mean)),size=15,transform=panel.transAxes)

        fig.suptitle(r'$\bf{CNTL:}$ ' + self.case1 + '\n' + r'$\bf{TEST:}$ ' + self.case2 + '\n' + r'$\bf{VRBL:}$ ' + self.var,
                     fontsize=20, horizontalalignment='left', x=0.125, y=0.96)
        ## Saving figure
        plt.savefig(str(self.path)+'/'+self.var+'_'+seasons[ind]+'_latlon_'+self.pval+'.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
        plt.close()