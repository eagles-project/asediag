import xarray as xr
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
from matplotlib.colors import ListedColormap

from src.utils.asediag_utils import rounding, gen_colbar_range, get_nearestlatlon
from src.utils.nclCols import amwg256_map, BlueWhiteOrangeRed_map


class GenVerticalData:
    def __init__(self, path, case, aer, **kwargs):
        self.path = path
        self.case = case
        self.aer = aer
        self.mod = kwargs.get('mod', 'eam')
        self.lats = kwargs.get('lats', None)
        self.lons = kwargs.get('lons', None)
        # Optional parameters
        self.var_vars = kwargs.get('var_vars', None)
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
        if self.aer=='num':
            factors['fact'] = factors['factaa']
        else:
            factors['fact'] = factors['factbb']
        
        return factors
    
    def get_variable_list(self):
        """
        Generate the list of variables to be used based on the type of aerosol.
        """
        vlist = list(self.data.variables.keys())
        if self.aer in self.gvars:
            var_avars = fnmatch.filter(vlist, self.aer)
            var_cvars = []
        else:
            var_avars = fnmatch.filter(vlist, f"{self.aer}_a?")
            var_cvars = fnmatch.filter(vlist, f"{self.aer}_c?")
        return var_avars + var_cvars
    
    def resample_data(self, vdata):
        """
        Resample the data to a specific lat/lon grid and average over levels.
        """
        ll = self.lat.round().values.tolist()
        all_ll = ll * len(self.data.lev)
        dd = vdata.to_dataframe().drop(columns=['season'])
        dd['lat'] = all_ll
        dd = dd.groupby(['lev', 'lat']).mean()
        dd = dd.rolling(2).mean().iloc[::2, :]
        return dd.to_xarray()
    
    def get_Vplots_data(self, ts):
        """
        Get data for vertical plots based on a given timestamp (season).
        """
        self.data, self.lon, self.lat = self._open_dataset(ts)
        if self.sv == None:
            self.factors = self.calculate_factors()
            self.var_vars = self.get_variable_list()
            vdata = self.data[self.var_vars] * self.factors['fact']
            vdata[self.aer] = vdata.to_array().sum('variable')
            vars_list = self.var_vars + [self.aer]
        else:
            print('Extra variables:', self.aer)
            vlist = list(self.data.variables.keys())
            existing_vars_list = [item for item in self.aer if item in vlist]
            vars_list = [var for var in existing_vars_list if 'lev' in self.data[var].dims]
            NAvars = set(self.aer) - set(vars_list)
            if NAvars:
                print('\nThe following variables with vertical dimension are unavailable:\n',NAvars,'\n')
            vdata = self.data[vars_list]
        
        if self.lats is not None and self.lons is not None:
            lts, lns = [], []
            for lat1, lon1 in zip(self.lats, self.lons):
                lt1, _, ln1, _ = get_nearestlatlon(lon1, lat1, self.lon, self.lat)
                lts.append(lt1)
                lns.append(ln1)
        
            pdata = []
            for ln, lt in zip(lns, lts):
                var1 = vdata.where((self.lat == lt) & (self.lon == ln)).copy()
                var1 = var1.stack(grid=var1.dims).dropna("grid", how="all")
                levels = var1.lev.values
                var1 = var1.drop_vars('lev').assign_coords(grid=levels)
                pdata.append(var1)
        
            local_data = xr.concat(pdata, dim="location")
        else:
            local_data = None
    
        if 'ncol' in self.data.dims:
            vdata = self.resample_data(vdata.copy())
        else:
            vdata = vdata.mean(dim='lon')
    
        return vdata, vars_list, local_data
    
    def gather_ProfData(self):
        """
        Gather profile data for all specified seasons.
        """
        dlist, plist = [], []

        for s in self.seasons:
            vdata, vars_list, pdata = self.get_Vplots_data(s)
            dlist.append(vdata)
            if pdata is not None:
                plist.append(pdata)

        data_combined = xr.concat(dlist, "season")
        
        if plist:
            pdata_combined = xr.concat(plist, "season")
        else:
            pdata_combined = None

        return data_combined, vars_list, pdata_combined


class GetVerticalProfiles:
    def __init__(self, data1, data2, diff, rel, var, ind, case1, case2, path=None, gunit=None, loc=None):
        self.loc = loc
        if self.loc == None:
            self.data1 = data1.isel(season=ind)
            self.data2 = data2.isel(season=ind)
            self.diff = diff.isel(season=ind)
            self.rel = rel.isel(season=ind)
        else:
            self.data1 = data1
            self.data2 = data2
            self.diff = diff
            self.rel = rel
        self.var = var
        self.ind = ind
        self.case1 = case1
        self.case2 = case2
        self.path = path
        self.gunit = gunit or self._get_unit(var)
        self.titles = ['Control Case', 'Test Case', 'Test Case - Control Case', 'Relative diff (%)']
        self.colBars = [
            gen_colbar_range(v1=self.data1, v2=self.data2).vmap(),
            gen_colbar_range(v1=self.data1, v2=self.data2).vmap(),
            gen_colbar_range(diff=self.diff, v1=self.data1).vdiff(),
            [-100., -50., -20., -10., -5., -2., 2., 5., 10., 20., 50., 100.]
        ]
        self.colMaps = [amwg256_map, amwg256_map, BlueWhiteOrangeRed_map, BlueWhiteOrangeRed_map]
        self.units = [self.gunit, self.gunit, self.gunit, '[%]']
        self.varbls = [self.data1, self.data2, self.diff, self.rel]
        self.seasons = ['ANN', 'DJF', 'JJA']

    def _get_unit(self, var):
        """
        Get the unit for the given variable.
        """
        if 'num' in var:
            return '[# cm$^{-3}$]'
        else:
            return '[ug m$^{-3}$]'

    @staticmethod
    def _get_cmap(cm, cbs, cbe, cbi):
        """
        Get a colormap.
        """
        try:
            return ListedColormap(cm.colors[cbs:cbe:cbi])
        except:
            print('Cannot subscript Segmented Colormap!')
            return cm

    def plot(self):
        """
        Plot 4=panel vertical profiles.
        """
        fig = plt.figure(figsize=(18, 14))

        for i, (title, colr, unit, cmap, vals) in enumerate(zip(self.titles, self.colBars, self.units, self.colMaps, self.varbls), 1):
            cbs, cbe, cbi = self._get_cmap_params(i)
            ax = plt.subplot(220 + i)
            self._plot_single_panel(ax, vals, colr, unit, cmap, cbs, cbe, cbi)
            ax.text(0.005, 1.03, title, size=15, transform=ax.transAxes)

        fig.suptitle(
            f'$\\bf{{Control\\ Case:}}$ {self.case1}\n$\\bf{{Test\\ Case:}}$ {self.case2}\n$\\bf{{Plotting:}}$ {self.var}',
            fontsize=20, horizontalalignment='left', x=0.125, y=0.98
        )

        plt.savefig(f'{self.path}/{self.var}_{self.seasons[self.ind]}_lathgt.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    def local_plot(self):
        """
        Plot local vertical profiles.
        """
        fig = plt.figure(figsize=(14, 14))
        for i, (title, unit, vals) in enumerate(zip(self.titles, self.units, self.varbls), 1):
            ax = plt.subplot(220 + i)
            self._plot_local_panel(ax, vals, unit)
            ax.text(0.005, 1.03, title, size=15, transform=ax.transAxes)
            if i==1:
                plt.legend(fontsize=12)

        fig.suptitle(
            f'$\\bf{{Control\\ Case:}}$ {self.case1}\n$\\bf{{Test\\ Case:}}$ {self.case2}\n$\\bf{{Plotting:}}$ {self.var}',
            fontsize=20, horizontalalignment='left', x=0.125, y=0.98
        )

        plt.savefig(f'{self.path}/{self.var}_{self.loc}_lathgt.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)


    def _get_cmap_params(self, panel_index):
        """
        Get colormap parameters to control the colorbar.
        """
        if panel_index < 3:
            return (5, -20, 2) if (np.max(self.data1).values + np.max(self.data2).values) / 2 != 0 else (-20, 5, -2)
        else:
            return (0, -1, 5) if (np.max(self.data1).values + np.max(self.data2).values) / 2 != 0 else (-1, 0, -5)

    def _plot_single_panel(self, ax, data, ranges, unit, cm, cbs, cbe, cbi):
        """
        Plot a single panel of vertical profile data.
        """
        cmap = self._get_cmap(cm, cbs, cbe, cbi)
        x, y = np.meshgrid(data['lat'], data['lev'])
        clen = len(np.arange(0, 257)[cbs:cbe:cbi])

        im = ax.contourf(
            x, y, data[:], cmap=cmap, levels=ranges,
            norm=matplotlib.colors.BoundaryNorm(boundaries=ranges, ncolors=clen), extend='both'
        )
        ax.invert_yaxis()
        ax.set_xlim([-89, 88])
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.set_xticklabels(['60S', '30S', '0', '30N', '60N'], size=12)
        ax.yaxis.set_tick_params(width=1.5, length=5)
        ax.xaxis.set_tick_params(width=1.5, length=5)
        ax.grid(lw=0.5, color='#EBE7E0', alpha=0.5, linestyle='-.')
        ax.tick_params(labelsize=12)

        cbar = plt.colorbar(im, ticks=ranges, drawedges=True, extendrect=True)
        cbar_ticks = self._format_colorbar_ticks(ranges)
        cbar.ax.set_yticklabels(cbar_ticks, size=12)
        cbar.set_label(label=unit, size=12)
        cbar.outline.set_linewidth(1.5)
        cbar.dividers.set_linewidth(1.5)
        plt.setp(ax.spines.values(), lw=1.5)
        plt.ylabel('Pressure [hPa]', fontsize=15)
    
    @staticmethod
    def _plot_local_panel(ax, data, unit):
        """
        Plot a local panel of vertical profile data.
        """
        plt.plot(data.isel(season=0),data['grid'],color=(0.8705882352941177, 0.5607843137254902, 0.0196078431372549),label='ANN')
        plt.plot(data.isel(season=1),data['grid'],color='gray',label='DJF')
        plt.plot(data.isel(season=2),data['grid'],color='gray',label='JJA',linestyle='--')
        ranges = list(plt.xticks()[0])
        s1 = pd.DataFrame(ranges)
        s2 = s1.map(lambda x: rounding(x))[0].tolist()
        cbar_ticks=list(map(str,s2))
        plt.xticks(ranges,cbar_ticks)
        plt.gca().invert_yaxis()
        ax.yaxis.set_tick_params(width=1.5,length=5)
        ax.xaxis.set_tick_params(width=1.5,length=5)
        ax.grid( lw=0.5, color='#EBE7E0', alpha=0.5, linestyle='-.')
        ax.tick_params(labelsize=12)
        plt.setp(ax.spines.values(),lw=1.5)
        plt.ylabel('Pressure [hPa]',fontsize=15)
        plt.xlabel(unit,fontsize=15)

    @staticmethod
    def _format_colorbar_ticks(ranges):
        """
        Format colorbar ticks to remove unnecessary decimal points.
        """
        s1 = pd.DataFrame(ranges)
        s2 = s1.map(lambda x: rounding(x))[0].tolist()
        return [str(i).replace('.0', '') if str(i)[-2:] == '.0' else str(i) for i in s2]