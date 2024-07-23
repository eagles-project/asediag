import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import cartopy.crs as crs

from src.utils.nclCols import BlueWhiteOrangeRed_map
from src.utils.aerdiag_plots import get_plots
from src.utils.asediag_utils import rounding, get_latlon, get_nearestlatlon, get_local
from src.utils.html_utils import get_html_table

class ForcingAnalyzer:
    def __init__(self, path1, path2, case1, case2, path, season='ANN', mod='eam', scrip='northamericax4v1pg2_scrip.nc'):
        self.path1 = path1
        self.path2 = path2
        self.case1 = case1
        self.case2 = case2
        self.path = path
        self.season = season
        self.mod = mod
        self.scrip = scrip

    def forcing_plots(self, plot_vars, area, plane, lon1, lon2, lat1, lat2):
        """
        Generate and save forcing plots.
        """
        titles = ['TOA $\u0394$F : ALL', 'TOA $\u0394$F$_{SW}$ : ALL', 'TOA $\u0394$F$_{LW}$ : ALL',
                  'TOA $\u0394$F : IND', 'TOA $\u0394$F$_{SW}$ : IND', 'TOA $\u0394$F$_{LW}$ : IND',
                  'TOA $\u0394$F : DIR', 'TOA $\u0394$F$_{SW}$ : DIR', 'TOA $\u0394$F$_{LW}$ : DIR',
                  'TOA $\u0394$F : clear-sky DIR', 'TOA $\u0394$F$_{SW}$ : clear-sky DIR', 'TOA $\u0394$F$_{LW}$ : clear-sky DIR',
                  'TOA $\u0394$F : RES', 'TOA $\u0394$F$_{SW}$ : RES', 'TOA $\u0394$F$_{LW}$ : RES']
        labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)',
                  '(k)', '(l)', '(m)', '(n)', '(o)']
        rr = [-20., -10., -5., -2., -1., -0.5, 0.5, 1., 2., 5., 10., 20.]
        
        plt.figure(figsize=(18, 16))
        i = 1
        for var, t, l in zip(plot_vars, titles, labels):
            ax = plt.subplot(5, 3, 0 + i, projection=crs.PlateCarree())
            m = (var * area).sum(area.dims) / area.sum(area.dims)
            try:
                get_plots(var, ax=ax, cmap=BlueWhiteOrangeRed_map, levels=rr,
                          scrip_file=self.scrip, gridLines=False,
                          lon_range=[lon1, lon2], lat_range=[lat1, lat2],
                          unit='[W m$^{-2}$]').get_map()
            except:
                get_plots(var, ax=ax, cmap=BlueWhiteOrangeRed_map, levels=rr,
                          scrip_file='', gridLines=False,
                          lon_range=[lon1, lon2], lat_range=[lat1, lat2],
                          unit='[W m$^{-2}$]').get_map()
            ax.text(0.005, 1.03, t, size=12, transform=ax.transAxes)
            ax.text(0.89, 1.03, '{:0.3f}'.format(m.values), size=12, transform=ax.transAxes)
            ax.text(0.05, 0.95, l, size=12, transform=ax.transAxes, va='top', bbox={'facecolor': 'white', 'pad': 1, 'edgecolor': 'none'})
            i += 1
        plt.savefig(str(self.path) + '/Forcing_' + plane + '_latlon.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    def get_forcings(self, datadef, datase, lon, lat, area, reg=None, loc=None):
        """
        Calculate and return the forcings.
        """
        if reg is not None:
            lat1, lat2, lon1, lon2 = get_latlon(reg)
        elif loc is not None:
            lat1, lon1 = get_local(loc)
            lat1, lat2, lon1, lon2 = get_nearestlatlon(lon1, lat1, lon, lat)
        else:
            lat1, lat2, lon1, lon2 = lat.values.min(), lat.values.max(), lon.values.min(), lon.values.max()
        # Calculate shortwave (SW) and longwave (LW) forcings for different components
        SWIND = (datase['FSNT_d1'] - datase['FSNTC_d1']) - (datadef['FSNT_d1'] - datadef['FSNTC_d1'])
        SWDIR = (datase['FSNT'] - datase['FSNT_d1']) - (datadef['FSNT'] - datadef['FSNT_d1'])
        SWDIR_C = (datase['FSNTC'] - datase['FSNTC_d1']) - (datadef['FSNTC'] - datadef['FSNTC_d1'])
        SWALB = datase['FSNTC_d1'] - datadef['FSNTC_d1']
        LWIND = -1 * ((datase['FLNT_d1'] - datase['FLNTC_d1']) - (datadef['FLNT_d1'] - datadef['FLNTC_d1']))
        LWDIR = -1 * ((datase['FLNT'] - datase['FLNT_d1']) - (datadef['FLNT'] - datadef['FLNT_d1']))
        LWALB = -1 * (datase['FLNTC_d1'] - datadef['FLNTC_d1'])
        AIND = SWIND + LWIND
        ADIR = SWDIR + LWDIR
        AALB = SWALB + LWALB
        AA = AIND + ADIR + AALB
        TTAEF = -1 * (datadef['FSNT'] - datadef['FLNT'] - (datase['FSNT'] - datase['FLNT']))
        SWAEF = -1 * (datadef['FSNT'] - datase['FSNT'])
        LWAEF = (datadef['FLNT'] - datase['FLNT'])
        SWCAEF = -1 * (datadef['FSNTC'] - datase['FSNTC'])
        SWCAEF_clean = -1 * (datadef['FSNTC_d1'] - datase['FSNTC_d1'])
        LWCAEF = (datadef['FLNTC'] - datase['FLNTC'])
        LWCAEF_clean = 1 * (datadef['FLNTC_d1'] - datase['FLNTC_d1'])
        TTIND = SWIND + LWIND
        TTDIR = SWDIR + LWDIR
        SWCDIR = (SWCAEF - SWCAEF_clean)
        LWCDIR = (LWCAEF - LWCAEF_clean)
        TTCDIR = SWCDIR + LWCDIR
        TTALB = SWALB + LWALB
        # Generate plots if the region is Global
        if reg == 'Global':
            plot_vars = [TTAEF, SWAEF, LWAEF, TTIND, SWIND, LWIND, TTDIR, SWDIR, LWDIR,
                         TTCDIR, SWCDIR, LWCDIR, TTALB, SWALB, LWALB]
            names = ['TTAEF', 'SWAEF', 'LWAEF', 'TTIND', ' SWIND', 'LWIND', 'TTDIR', 'SWDIR', 'LWDIR', 'TTCDIR', 'SWCDIR', 'LWCDIR', 'TTALB', 'SWALB', 'LWALB']
            for n, v in zip(names, plot_vars):
                v.name = n.strip()
            saving_data = xr.merge(plot_vars)
            # saving_data.load().to_netcdf(self.path + '/TOA_forcing_vars_global.nc')
            if self.season == 'ANN':
                self.forcing_plots(plot_vars, area, 'TOA', lon1, lon2, lat1, lat2)
        
        # Calculate regional means for all forcings
        all_vars = [TTAEF, SWAEF, LWAEF, SWCAEF, LWCAEF, SWIND, LWIND, TTIND, SWDIR, LWDIR, TTDIR,
                    SWCDIR, LWCDIR, TTCDIR, SWALB, LWALB, TTALB]
        all_means = []
        for vdata in all_vars:
            vdatalatlon = vdata.where((lon >= lon1) & (lon <= lon2) & (lat >= lat1) & (lat <= lat2))
            arealatlon = area.where((lon >= lon1) & (lon <= lon2) & (lat >= lat1) & (lat <= lat2))
            mean = (vdatalatlon * arealatlon).sum(arealatlon.dims) / (arealatlon).sum(arealatlon.dims)
            all_means.append(mean.values)
        # Repeat the above calculations for surface forcing  
        SWIND = (datase['FSNS_d1'] - datase['FSNSC_d1']) - (datadef['FSNS_d1'] - datadef['FSNSC_d1'])
        SWDIR = (datase['FSNS'] - datase['FSNS_d1']) - (datadef['FSNS'] - datadef['FSNS_d1'])
        SWDIR_C = (datase['FSNSC'] - datase['FSNSC_d1']) - (datadef['FSNSC'] - datadef['FSNSC_d1'])
        SWALB = datase['FSNSC_d1'] - datadef['FSNSC_d1']
        LWIND = -1 * ((datase['FLNS_d1'] - datase['FLNSC_d1']) - (datadef['FLNS_d1'] - datadef['FLNSC_d1']))
        LWDIR = -1 * ((datase['FLNS'] - datase['FLNS_d1']) - (datadef['FLNS'] - datadef['FLNS_d1']))
        LWALB = -1 * (datase['FLNSC_d1'] - datadef['FLNSC_d1'])
        AIND = SWIND + LWIND
        ADIR = SWDIR + LWDIR
        AALB = SWALB + LWALB
        AA = AIND + ADIR + AALB
        TTAEF = -1 * (datadef['FSNS'] - datadef['FLNS'] - (datase['FSNS'] - datase['FLNS']))
        SWAEF = -1 * (datadef['FSNS'] - datase['FSNS'])
        LWAEF = (datadef['FLNS'] - datase['FLNS'])
        SWCAEF = -1 * (datadef['FSNSC'] - datase['FSNSC'])
        SWCAEF_clean = -1 * (datadef['FSNSC_d1'] - datase['FSNSC_d1'])
        LWCAEF = (datadef['FLNSC'] - datase['FLNSC'])
        LWCAEF_clean = 1 * (datadef['FLNSC_d1'] - datase['FLNSC_d1'])
        TTIND = SWIND + LWIND
        TTDIR = SWDIR + LWDIR
        SWCDIR = (SWCAEF - SWCAEF_clean)
        LWCDIR = (LWCAEF - LWCAEF_clean)
        TTCDIR = SWCDIR + LWCDIR
        TTALB = SWALB + LWALB
    
        if reg == 'Global':
            plot_vars = [TTAEF, SWAEF, LWAEF, TTIND, SWIND, LWIND, TTDIR, SWDIR, LWDIR,
                         TTCDIR, SWCDIR, LWCDIR, TTALB, SWALB, LWALB]
            names = ['TTAEF', 'SWAEF', 'LWAEF', 'TTIND', ' SWIND', 'LWIND', 'TTDIR', 'SWDIR', 'LWDIR', 'TTCDIR', 'SWCDIR', 'LWCDIR', 'TTALB', 'SWALB', 'LWALB']
            for n, v in zip(names, plot_vars):
                v.name = n.strip()
            saving_data = xr.merge(plot_vars)
            # saving_data.load().to_netcdf(str(self.path) + '/SFC_forcing_vars_global.nc')
            if self.season == 'ANN':
                self.forcing_plots(plot_vars, area, 'SFC', lon1, lon2, lat1, lat2)
    
        all_vars = [TTAEF, SWAEF, LWAEF, SWCAEF, LWCAEF, SWIND, LWIND, TTIND, SWDIR, LWDIR, TTDIR,
                    SWCDIR, LWCDIR, TTCDIR, SWALB, LWALB, TTALB]
        for vdata in all_vars:
            vdatalatlon = vdata.where((lon >= lon1) & (lon <= lon2) & (lat >= lat1) & (lat <= lat2))
            arealatlon = area.where((lon >= lon1) & (lon <= lon2) & (lat >= lat1) & (lat <= lat2))
            mean = (vdatalatlon * arealatlon).sum(arealatlon.dims) / (arealatlon).sum(arealatlon.dims)
            all_means.append(mean.values)
        return all_means
    
    def get_forcing_df(self, regions=['Global', 'SH_pole', 'SH_midlat', 'Tropics', 'NH_midlat', 'NH_pole']):
        """
        Generate a dataframe with forcings for the specified regions and save as CSVs.
        """
        try:
            print('SE data:',f"{self.path1}{self.case1}.{self.mod}.{self.season}.*_climo.nc")
            file_path = f"{self.path1}{self.case1}.{self.mod}.{self.season}.*_climo.nc"
            datadef = xr.open_mfdataset(file_path)
            lon = datadef['lon']
            lon.load()
            lon[lon > 180.] -= 360.
        except:
            print('Lat/Lon data:',f"{self.path1}{self.case1}*{self.season}_*_climo.nc")
            file_path = f"{self.path1}{self.case1}*{self.season}_*_climo.nc"
            datadef = xr.open_mfdataset(file_path)
            if 'time' in datadef.dims:
                datadef = datadef.isel(time=0)
            lon = xr.where(datadef.lon > 180, datadef.lon - 360, datadef.lon)
            lon.load()
            lon = lon.assign_coords(lon=lon)
            datadef['lon'] = lon
            lon = lon.sortby(lon)
            datadef = datadef.sortby('lon')

        try:
            print('SE data:',f"{self.path2}{self.case2}.{self.mod}.{self.season}.*_climo.nc")
            file_path = f"{self.path2}{self.case2}.{self.mod}.{self.season}.*_climo.nc"
            datase = xr.open_mfdataset(file_path)
            lon = datase['lon']
            lon.load()
            lon[lon > 180.] -= 360.
        except:
            print('Lat/Lon data:',f"{self.path2}{self.case2}*{self.season}_*_climo.nc")
            file_path = f"{self.path2}{self.case2}*{self.season}_*_climo.nc"
            datase = xr.open_mfdataset(file_path)
            if 'time' in datase.dims:
                datase = datase.isel(time=0)
            lon = xr.where(datase.lon > 180, datase.lon - 360, datase.lon)
            lon.load()
            lon = lon.assign_coords(lon=lon)
            datase['lon'] = lon
            lon = lon.sortby(lon)
            datase = datase.sortby('lon')
    
        lat = datase['lat']
    
        varlist = ['AODVIS', 'FSNT', 'FLNT', 'FSNTC', 'FLNTC', 'FSNT_d1', 'FLNT_d1',
                   'FSNTC_d1', 'FLNTC_d1', 'FSNS', 'FLNS', 'FSNSC', 'FLNSC', 'FSNS_d1',
                   'FLNS_d1', 'FSNSC_d1', 'FLNSC_d1']
        
        area = datase['area']
        ## This section handles missing variables so that the code doesn't break.
        dims = area.dims
        shape = area.shape
        def_vlist = list(datadef.variables.keys())
        existing_vars_def = [item for item in def_vlist if item in varlist]
        se_vlist = list(datase.variables.keys())
        existing_vars_se = [item for item in se_vlist if item in varlist]
        new_varlist = [item for item in existing_vars_se if item in existing_vars_def]
        NAvars = list(set(varlist) - set(new_varlist))
        if NAvars:
            print('\nThe following variables are unavailable:\n',NAvars,'\nThese will be added as dummy NaN variables.\n')

        datadef = datadef[new_varlist]
        datase = datase[new_varlist]

        # Iterate through the list to create dummy variables
        for var in NAvars:
            datadef[var] = xr.DataArray(np.nan * np.ones(shape), dims=dims)
            datase[var] = xr.DataArray(np.nan * np.ones(shape), dims=dims)

        lon[lon > 180.] -= 360.
        var_names = ['TTAEF', 'SWAEF', 'LWAEF', 'SWCAEF', 'LWCAEF', 'SWIND', 'LWIND',
                     'TTIND', 'SWDIR', 'LWDIR', 'TTDIR', 'SWCDIR', 'LWCDIR', 'TTCDIR',
                     'SWALB', 'LWALB', 'TTALB', 'TTAEFs', 'SWAEFs', 'LWAEFs', 'SWCAEFs',
                     'LWCAEFs', 'SWINDs', 'LWINDs', 'TTINDs', 'SWDIRs', 'LWDIRs',
                     'TTDIRs', 'SWCDIRs', 'LWCDIRs', 'TTCDIRs', 'SWALBs', 'LWALBs',
                     'TTALBs']
        df = pd.DataFrame()
        for reg in regions:
            df[reg] = self.get_forcings(datadef, datase, lon, lat, area, reg=reg)
        df.index = var_names
        df.to_csv(str(self.path) + '/AllForcings_' + self.season + '.csv', index=False)
        pd.options.display.float_format = '{:g}'.format
        df = df.map(lambda x: rounding(x))
        df = df.astype(str)
        htable = get_html_table(df)
        with open(str(self.path) + '/' + 'Forcing_' + self.season + '_latlon.html', 'w') as f:
            f.write(htable)