import xarray as xr
import pandas as pd
import numpy as np
import fnmatch
import re

from asediag.asediag_utils import get_local, get_nearestlatlon, get_vertint, get_latlon

class AerosolBudgetCalculator:
    def __init__(self, path, case, ts, aer, **kwargs):
        self.path = path
        self.case = case
        self.ts = ts
        self.aer = aer
        self.mod = kwargs.get('mod', 'eam')
        
        # Load data
        self.data, self.lon = self.open_data()
        self.lat = self.data['lat']
        
        # Constants and factors
        self.avgod = 6.022e+23
        self.mwso4 = 115.0
        self.factors = self.calculate_factors()
        self.ps = self.data['PS']
        self.ha = self.data['hyai']
        self.hb = self.data['hybi']
        self.p0 = self.data['P0']
        self.area = self.data['area'] * (6.37122e6) ** 2
        self.landF = self.data['LANDFRAC']
        self.grav = self.factors["grav"]
        
        # Optional parameters
        self.reg = kwargs.get('reg', None)
        self.loc = kwargs.get('loc', None)
        self.indl = kwargs.get('indl', None)
        self.land = kwargs.get('land', False)
        self.splots = kwargs.get('splots', None)
        
        # Initial variables for SO2
        self.wdep_so2 = 0.0
        self.elev_emis_so2 = 0.0
        
        # Variables for output names
        self.bname = ''
        self.snkname = ''
        self.srcname = ''
        self.sname = ''
        self.vars1 = []
        self.gvars = kwargs.get('gvars', ['SO2', 'DMS', 'H2SO4', 'SOAG'])
        self.MAM_modes = {
                        '1': 'accum',
                        '2': 'aitken',
                        '3': 'coarse',
                        '4': 'pcarbon'
                        }

        # Sum of airmass for normalization
        self.sum_airmass = ((self.ps * self.area).sum() / self.grav) * 1e6

    def open_data(self):
        """Open and load dataset files."""
        try:
            print('SE data:',f"{self.path}{self.case}.{self.mod}.{self.ts}.*_climo.nc")
            file_path = f"{self.path}{self.case}.{self.mod}.{self.ts}.*_climo.nc"
            data = xr.open_mfdataset(file_path)
            lon = data['lon'].values
            lon[lon > 180.] -= 360.
        except:
            print('Lat/Lon data:',f"{self.path}{self.case}*{self.ts}_*_climo.nc")
            file_path = f"{self.path}{self.case}*{self.ts}_*_climo.nc"
            data = xr.open_mfdataset(file_path)
            if 'time' in data.dims:
                data = data.isel(time=0)
            lon = xr.where(data.lon > 180, data.lon - 360, data.lon)
            lon = lon.assign_coords(lon=lon.values)
            data['lon'] = lon
            lon = lon.sortby(lon)
            data = data.sortby('lon')
        
        if 'year' in data.coords:
            data = data.rename({'year':'season'})
            
        return data, lon

    def calculate_factors(self):
        """Calculate and return conversion factors."""
        factors = {
            "fact": 1e-9,
            "grav": 9.806,
            "factaa": self.mwso4 * 10.0 / self.avgod,        # convert molec/cm2/s to kg/m2/s
            "factaaa": 12.0 * 10.0 / self.avgod,             # convert molec/cm2/s to kg/m2/s
            "factbb": 86400.0 * 365.0 * 1e-9,                # convert kg/s to Tg/yr
            "factcc": None,
            "factdd": 32.066 / self.mwso4 * 1e-9,            # convert kg to TgS
            "fact_kgpkg_kgpcm3": 1.01325e5 / 8.31446261815324 / 273.15 * 28.9647 / 1.e9,  # kg-air/cm3-air
            "fact_kgpcm3_ugpm3": None,
        }
        if self.aer == 'num':
            factors["factbb"] = 86400.0 * 365.0
            factors["factaa"] = 1.e4 / (self.avgod * 1.e3)
        factors["factcc"] = factors["factbb"] / self.mwso4 * 32.066
        factors["fact_kgpcm3_ugpm3"] = factors["fact_kgpkg_kgpcm3"] * 1e15
        return factors

    def set_region(self):
        """Set the region for analysis."""
        if self.reg is not None:
            self.lat1, self.lat2, self.lon1, self.lon2 = get_latlon(self.reg)
        elif self.loc is not None:
            self.lat1, self.lon1 = get_local(self.loc)
            self.lat1, self.lat2, self.lon1, self.lon2 = get_nearestlatlon(self.lon1, self.lat1, self.lon, self.lat)
        else:
            self.lat1, self.lat2, self.lon1, self.lon2 = self.lat.values.min(), self.lat.values.max(), self.lon.min(), self.lon.max()

    def get_var_lists(self):
        avariables = [self.aer+'_a?',self.aer+'_a?'+'DDF',self.aer+'_a?'+'SFWET','SF'+self.aer+'_a?',self.aer+'_a?'+'_CLXF',\
             self.aer+'_a?'+'_sfgaex1',self.aer+'_a?'+'GVF',self.aer+'_a?'+'TBF',self.aer+'_a?'+'SFSIS',\
             self.aer+'_a?'+'SFSIC',self.aer+'_a?'+'SFSBS',self.aer+'_a?'+'SFSBC',self.aer+'_a?'+'SFSES',\
             self.aer+'_a?'+'SFSEC',self.aer+'_a?'+'_sfgaex2',self.aer+'_a?'+'_sfcoag1',self.aer+'_a?'+'_sfcsiz3',\
             self.aer+'_a?'+'_sfcsiz4',self.aer+'_a?'+'_mixnuc1',self.aer+'_a?'+'AQH2SO4',\
             self.aer+'_a?'+'AQSO4',self.aer+'_a?'+'_sfnnuc1','AQ_'+self.aer+'_a?','GS_'+self.aer+'_a?',self.aer+'_a?']
        cvariables = [self.aer+'_c?',self.aer+'_c?'+'DDF',self.aer+'_c?'+'SFWET','SF'+self.aer+'_c?',self.aer+'_c?'+'_CLXF',\
             self.aer+'_c?'+'_sfgaex1',self.aer+'_c?'+'GVF',self.aer+'_c?'+'TBF',self.aer+'_c?'+'SFSIS',\
             self.aer+'_c?'+'SFSIC',self.aer+'_c?'+'SFSBS',self.aer+'_c?'+'SFSBC',self.aer+'_c?'+'SFSES',\
             self.aer+'_c?'+'SFSEC',self.aer+'_c?'+'_sfgaex2',self.aer+'_c?'+'_sfcoag1',self.aer+'_c?'+'_sfcsiz3',\
             self.aer+'_c?'+'_sfcsiz4',self.aer+'_c?'+'_mixnuc1',self.aer+'_c?'+'AQH2SO4',\
             self.aer+'_c?'+'AQSO4',self.aer+'_c?'+'_sfnnuc1','AQ_'+self.aer+'_c?','GS_'+self.aer+'_c?',self.aer+'_c?']
        
        if self.aer in self.gvars:
            avariables = str(avariables).replace('_a?','').replace("'",'').replace(' ','')[1:-1].replace(self.aer+'DDF','DF_'+self.aer).replace(self.aer+'SFWET','WD_'+self.aer).split(',')
            cvariables = ['']*len(cvariables)
        
        return avariables, cvariables

    def adjust_vdata(self, vdata, avar, cvar, nvar, var_vars):
        if ((avar == self.aer + '_a?') and (nvar == 1)) or ((avar == self.aer) and (nvar == 1)):
            self.vars1 = var_vars+[avar+'+'+cvar]
            if self.aer == 'so4':
                self.bname = 'Burden (TgS)'
                self.srcname = 'Sources (TgS/yr)'
                self.snkname = 'Sinks (TgS/yr)'
                vdata = get_vertint(vdata, self.ha, self.p0, self.hb, self.ps, self.grav, self.factors['factdd'])
            elif self.aer == 'num':
                self.bname = 'Burden (#/mg-air)'
                self.srcname = 'Sources (#/mg-air/yr)'
                self.snkname = 'Sinks (#/mg-air/yr)'
                vdata = get_vertint(vdata, self.ha, self.p0, self.hb, self.ps, self.grav, 1) / self.sum_airmass
            else:
                self.bname = 'Burden (Tg)'
                self.srcname = 'Sources (Tg/yr)'
                self.snkname = 'Sinks (Tg/yr)'
                vdata = get_vertint(vdata, self.ha, self.p0, self.hb, self.ps, self.grav, self.factors['fact'])
        elif ((avar == self.aer+'_a?') and (nvar > 1)) or ((avar == self.aer) and (nvar > 1)):
            if self.aer == 'num':
                self.sname = 'Sfc Conc. (#/cm3)'
                vdata = vdata[dict(lev=-1)].drop_vars('lev')
                vdata = vdata*1
            else:
                self.sname = 'Sfc Conc. (ug/m3)'
                vdata = vdata[dict(lev=-1)].drop_vars('lev')
                vdata = vdata*self.factors['fact_kgpcm3_ugpm3']
        else:
            vdata = self.adjust_vdata_others(vdata, avar, cvar, nvar)
        return vdata

    def adjust_vdata_others(self, vdata, avar, cvar, nvar):
        if ('_CLXF' in avar):
            if self.aer == 'bc' or self.aer == 'pom':
                vdata = vdata * self.factors['factaaa'] * self.factors['factbb']
            elif self.aer == 'num':
                vdata = vdata * self.factors['factaa'] * self.factors['factbb'] / self.sum_airmass
            elif (self.aer == 'so4') or (self.aer in self.gvars):
                vdata = vdata * self.factors['factaa'] * self.factors['factcc']
            else:
                vdata = vdata * self.factors['factaa'] * self.factors['factbb']
        else:
            if (self.aer == 'so4') or (self.aer in self.gvars):
                vdata = self.adjust_gas_phase_species(vdata)
            elif self.aer == 'num':
                vdata = self.adjust_number_concentration(vdata)
            else:
                vdata = self.adjust_other_species(vdata)
        return vdata

    def adjust_gas_phase_species(self, vdata):
        if ('ncol' in self.data.dims) and (len(vdata.dims) > 1):
            vdata = get_vertint(vdata, self.ha, self.p0, self.hb, self.ps, self.grav, self.factors['factcc'])
        elif len(vdata.dims) > 2:
            vdata = get_vertint(vdata, self.ha, self.p0, self.hb, self.ps, self.grav, self.factors['factcc'])
        else:
            vdata = vdata * self.factors['factcc']
        return vdata

    def adjust_number_concentration(self, vdata):
        if ('ncol' in self.data.dims) and (len(vdata.dims) > 1):
            vdata = get_vertint(vdata, self.ha, self.p0, self.hb, self.ps, self.grav, self.factors['factbb']) / self.sum_airmass
        elif len(vdata.dims) > 2:
            vdata = get_vertint(vdata, self.ha, self.p0, self.hb, self.ps, self.grav, self.factors['factbb']) / self.sum_airmass
        else:
            vdata = vdata * self.factors['factbb'] / self.sum_airmass
        return vdata

    def adjust_other_species(self, vdata):
        if ('ncol' in self.data.dims) and (len(vdata.dims) > 1):
            vdata = get_vertint(vdata, self.ha, self.p0, self.hb, self.ps, self.grav, self.factors['factbb'])
        elif len(vdata.dims) > 2:
            vdata = get_vertint(vdata, self.ha, self.p0, self.hb, self.ps, self.grav, self.factors['factbb'])
        else:
            vdata = vdata * self.factors['factbb']
        return vdata

    def calculate_mean(self, vdata, avar):
        """Calculate area weighted sums."""
        if self.indl is not None:
            try:
                mean = (vdata.sel(ncol=self.indl)).mean(dim=['ncol'])
            except:
                vdatalatlon = vdata.where((self.lon>=self.lon1) & (self.lon<=self.lon2))
                vdatalatlon = vdatalatlon.where((self.lat>=self.lat1) & (self.lat<=self.lat2))
                arealatlon = self.area.where((self.lon>=self.lon1) & (self.lon<=self.lon2))
                arealatlon = arealatlon.where((self.lat>=self.lat1) & (self.lat<=self.lat2))
                mean = (vdatalatlon*arealatlon).sum(arealatlon.dims)
        elif ('WD_' in avar):
            vdatalatlon = vdata.where((self.lon>=self.lon1) & (self.lon<=self.lon2))
            vdatalatlon = vdatalatlon.where((self.lat>=self.lat1) & (self.lat<=self.lat2))
            arealatlon = self.area.where((self.lon>=self.lon1) & (self.lon<=self.lon2))
            arealatlon = arealatlon.where((self.lat>=self.lat1) & (self.lat<=self.lat2))
            mean = (vdatalatlon).sum(arealatlon.dims)
        else:
            vdatalatlon = vdata.where((self.lon>=self.lon1) & (self.lon<=self.lon2))
            vdatalatlon = vdatalatlon.where((self.lat>=self.lat1) & (self.lat<=self.lat2))
            arealatlon = self.area.where((self.lon>=self.lon1) & (self.lon<=self.lon2))
            arealatlon = arealatlon.where((self.lat>=self.lat1) & (self.lat<=self.lat2))
            mean = (vdatalatlon*arealatlon).sum(arealatlon.dims)
        return mean

    def append_to_dataframe(self, df, mean, avar, cvar, prob_list):
        """Append calculated sums to the dataframe."""
        rvars = dict(zip(prob_list+[avar+'+'+cvar],self.vars1))
        mean = mean.rename_vars(rvars)
        if ('DDF' in avar) or ('GVF' in avar) or ('TBF' in avar) or ('DF_' in avar) or ('WD_' in avar):
            mean = -1 * mean
        ### Treat SO2 separately
        if self.aer == 'SO2':
            if 'WD_' in avar:
                self.wdep_so2 = mean
            elif '_CLXF' in avar:
                self.elev_emis_so2 = mean
            # GS_SO2 = SO2 emission (elevated) + WD_SO2 + Chemical reactions
            elif 'GS_' in avar:
                mean = mean + self.wdep_so2 - self.elev_emis_so2
                
        ## Appending to dataframe
        ndf = mean.expand_dims(dim='vars').to_dataframe()
        df = pd.concat([df,ndf.replace(0, np.nan)])
        return df

    def calculate_totals(self, vdata, avar, cvar, var_vars):
        """Calculate totals for the data and handle placeholder variables."""
        prob_list=[]
        for item in self.vars1[:-1]:
            prob_list.append(avar.replace(self.aer+'_a?',item))
        unavail_vars = list(set(prob_list)-set(var_vars+[avar+'+'+cvar]))
        if var_vars!=[]:
            vdata[avar+'+'+cvar] = vdata.to_array().sum('variable')
        else:
            vdata[avar+'+'+cvar] = np.nan
        if unavail_vars!=[]:
            vdata[unavail_vars] = [np.nan]*len(unavail_vars)
        
        if self.land==True:
            vdata = vdata.where(self.landF>0)
        else:
            vdata = vdata.where(self.landF>=0)
        return vdata, prob_list

    def process_data(self, avariables, cvariables):
        """Process data to generate a dataframe and a list of variables data."""
        df = pd.DataFrame()
        nvar = 0
        nvdata = []
        for avar, cvar in zip(avariables[:], cvariables[:]):
            nvar += 1
            var_avars = fnmatch.filter(self.data.variables.keys(), avar)
            var_cvars = fnmatch.filter(self.data.variables.keys(), cvar)
            var_vars = var_avars + var_cvars
            vdata = self.data[var_vars]
            vdata = self.adjust_vdata(vdata, avar, cvar, nvar, var_vars)
            vdata, prob_list = self.calculate_totals(vdata, avar, cvar, var_vars)
            nvdata.append(vdata)
            mean = self.calculate_mean(vdata, avar)
            df = self.append_to_dataframe(df, mean, avar, cvar, prob_list)
        return df, nvdata

    def save_to_netcdf(self, nvdata):
        """Save processed data to NetCDF files for spatial map plots of the budget components."""
        print(f'\nSaving all budget table variables for {self.aer}')
        newVdata = xr.merge(nvdata, compat='override')
        newVdata.load().to_netcdf(f"{self.path}{self.case}_{self.aer}_{self.ts}_allVdata.nc")

    def finalize_dataframe(self, df):
        """Finalize the dataframe by adding necessary columns and calculating Lifetimes."""
        if 'ncol' in self.data.dims:    
            df['year'] = np.nan
            df = df[['season','year']+self.vars1]
        else:
            df['time'] = np.nan
            df = df[['time']+self.vars1]

        index_list = [self.bname,'Dry deposition','Wet deposition','surface emission',\
             'elevated emission','condensation-aging','gravitational','turbulent',\
             'incloud, stratiform','incloud, convective','belowcloud, strat.',\
              'belowcloud, convec.','rain evap, strat.','rain evap, convec.',\
             'renaming (sfgaex2)','coagulation (sfcoag1)','calcsize (sfcsiz3)',\
             'calcsize (sfcsiz4)','dropmixnuc (mixnuc1)','cloudchem (AQH2SO4)',\
             'cloudchem (AQSO4)','sfnnuc1','Aq. chem (gas-species)','gas chem/wet dep. (gas-species)',self.sname]
        df.index=index_list
        
        listofSS = ['Dry deposition','Wet deposition','renaming (sfgaex2)',\
                     'coagulation (sfcoag1)','calcsize (sfcsiz3)',\
                     'calcsize (sfcsiz4)','dropmixnuc (mixnuc1)',\
                     'condensation-aging','surface emission','elevated emission',\
                     'cloudchem (AQH2SO4)','cloudchem (AQSO4)','sfnnuc1',\
                     'Aq. chem (gas-species)','gas chem/wet dep. (gas-species)']
        
        if self.aer in self.gvars:
            self.aer = 'total_'+self.aer
            df.columns=df.columns.tolist()[:-1]+[self.aer]
            srcsnk = df.loc[listofSS][self.vars1[:-1]+[self.aer]]
        else:
            df.columns=df.columns.tolist()[:-1]+[self.aer]
            srcsnk = df.loc[listofSS[:-2]][self.vars1[:-1]+[self.aer]]

        ## Estimating sources and sinks
        src = srcsnk.where(srcsnk>0).sum()
        if 'SO2' in self.aer:
            snk = df.loc['gas chem/wet dep. (gas-species)'] + df.loc['Dry deposition']
        else:
            snk = srcsnk.where(srcsnk<0).sum()

        df_copy = df.copy()
        df_copy.loc[self.srcname] = src
        df_copy.loc[self.snkname] = snk
        lifetime = (df_copy.loc[self.bname][self.vars1[:-1]+[self.aer]]/abs(df_copy.loc[self.snkname][self.vars1[:-1]+[self.aer]]))*365
        df_copy.loc['Lifetime (days)'] = lifetime
        
        if 'ncol' in self.data.dims:
            df_copy['season']=self.ts
        else:
            df_copy['time']=self.ts
        df = df_copy.reindex([self.bname,self.sname,self.srcname,'surface emission','elevated emission',self.snkname,\
               'Dry deposition','gravitational','turbulent','Wet deposition',\
               'incloud, stratiform','incloud, convective','belowcloud, strat.',\
               'belowcloud, convec.','rain evap, strat.','rain evap, convec.',\
               'Lifetime (days)','renaming (sfgaex2)','coagulation (sfcoag1)','calcsize (sfcsiz3)',\
                'calcsize (sfcsiz4)','dropmixnuc (mixnuc1)','cloudchem (AQH2SO4)',\
                'cloudchem (AQSO4)','condensation-aging','sfnnuc1','Aq. chem (gas-species)','gas chem/wet dep. (gas-species)'])
        
        return df
        
    def get_tables(self):
        """Generate and return the aerosol budget tables."""
        self.set_region()
        avariables, cvariables = self.get_var_lists()
        df, nvdata = self.process_data(avariables, cvariables)
        if self.splots is not None:
            self.save_to_netcdf(nvdata)
        df = self.finalize_dataframe(df)

        # Create a dictionary to store sums of columns for each aerosol mode
        aerosol_groups = {amode: [] for amode in self.MAM_modes.values()}

        # Find Pairs for each mode
        pattern = re.compile(f"{self.aer}_[ac](\d)")
        for col in df.columns:
            match = pattern.match(col)
            if match:
                suffix = match.group(1)
                if suffix in self.MAM_modes:
                    amode = self.MAM_modes[suffix]
                    aerosol_groups[amode].append(col)

        # Sum up the values for each mode
        for amode, columns in aerosol_groups.items():
            if columns:
                df[self.aer+'_'+amode] = df[columns].sum(axis=1)
        
        # Re-calculate lifetime for each mode
        updated_LT = ['ANN',np.nan]+list(((df.iloc[0][2:]/abs(df.iloc[5][2:]))*365).values)
        LT_series = pd.Series(updated_LT, index=df.columns)
        df.loc['Lifetime (days)'] = LT_series
        
        df = df.copy().loc[:, (df != 0).any(axis=0)]
        df = df.replace(0, np.nan)
        return df