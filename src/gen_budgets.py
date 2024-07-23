import fnmatch
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as crs
import matplotlib.gridspec as gridspec

from src.aer_budget_analysis import AerosolBudgetCalculator
from src.utils.asediag_utils import rounding, rearrange_variables
from src.utils.html_utils import get_html_table
from src.utils.nclCols import amwg256_map, BlueWhiteOrangeRed_map
from src.utils.aerdiag_plots import get_plots

class GenAerosolBudgets:
    def __init__(self, ind, aer, path1, path2, case1, case2, path, reg, loc, mod, land, splot):
        self.ind = ind
        self.aer = aer
        self.path1 = path1
        self.path2 = path2
        self.case1 = case1
        self.case2 = case2
        self.path = path
        self.reg = reg
        self.loc = loc
        self.mod = mod
        self.land = land
        self.splot = splot
        self.ss = ['ANN', 'DJF', 'JJA']
    
    def preprocess_data(self, data):
        data = data.reset_index()
        data.columns = ['metric', 'cntl', 'test', 'diff', 'rel']
        data = data.drop(index=[2, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 19, 20, 27]).reset_index(drop=True)
        metrics = [
        'Burden', 'Sfc Conc.', 'Surface\nemission', 'Elevated\nemission', 
        'Dry\ndeposition', 'Wet\ndeposition', 'Renaming', 'Coagulation', 
        'Dropmix\nnuc', 'Cloud\nchemistry\n(AQH2SO4)', 'Cloud\nchemistry\n(AQSO4)', 
        'Condensation', 'NPF', 'Aquous\nchemistry\n(gas-species)'
        ]
        data['metric'] = metrics
        return data

    def filter_data(self, data):
        data_sink = data[data['cntl'] < 0].sort_values(by='diff', ascending=False).reset_index(drop=True)
        selrel_sink = (data_sink['diff'] / abs(data_sink['diff']).max()) * 100
        data_sink = data_sink[abs(selrel_sink) >= 1].reset_index(drop=True)
        data_sink['diff'] = np.sign(data_sink['cntl']) * data_sink['diff']
        
        data_dropped = data.drop(index=[0, 1])
        data_source = data_dropped.loc[data['cntl'] > 0].sort_values(by='diff', ascending=True).reset_index(drop=True)
        selrel_source = (data_source['diff'] / abs(data_source['diff']).max()) * 100
        data_source = data_source[abs(selrel_source) >= 1].reset_index(drop=True)
        data_source['diff'] = np.sign(data_source['cntl']) * data_source['diff']
        
        return data_source, data_sink

    def gen_budget_barPlots(self, data, var, unit='(#/mg-air/yr)'):
        data = self.preprocess_data(data)
        data_source, data_sink = self.filter_data(data)
        
        fig = plt.figure(figsize=[14, 8])
        gs = gridspec.GridSpec(1, 5, wspace=0.2)
        
        # Plot sources and sinks
        ax1 = fig.add_subplot(gs[:, :4])
        xx = data_source['metric'].tolist() + data_sink['metric'].tolist()
        ax1.bar(np.arange(0, len(data_source)), data_source['diff'], color='gray', edgecolor='k', zorder=4, label='Sources')
        ax1.bar(np.arange(len(data_source), len(data_source) + len(data_sink)), data_sink['diff'], color='gray', hatch='//', edgecolor='k', zorder=4, label='Sinks')
        ax1.set_xticks(np.arange(0, len(data_source) + len(data_sink)), xx)
        ax1.grid(linestyle='--', color='#EBE7E0', zorder=4)
        ax1.set_axisbelow(True)
        ax1.set_ylabel(f'$\\Delta$({var})\n{unit}', fontsize=20)
        ax1.tick_params(labelsize=12)
        plt.setp(ax1.spines.values(), lw=1.5)
        ax1.tick_params(axis='x', which='both', bottom=False)
        plt.axhline(0, c='k')
        ax1.legend(fontsize=12)
        
        # Plot percent change
        ax2 = fig.add_subplot(gs[:, 4:])
        ax2.bar(0, data.iloc[0]['rel'], color='lightgray', width=0.5, edgecolor='k', zorder=4)
        ax2.bar(1, data.iloc[1]['rel'], color='lightgray', width=0.5, edgecolor='k', zorder=4)
        ax2.set_xticks([-0.5, 0, 1, 1.5], ['', 'Burden', 'Surface\nconc.', ''])
        ax2.grid(linestyle='--', color='#EBE7E0', zorder=4)
        ax2.set_axisbelow(True)
        ax2.set_ylabel('Percent change (%)', fontsize=20)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.tick_params(labelsize=12)
        plt.setp(ax2.spines.values(), lw=1.5)
        ax2.tick_params(axis='x', which='both', bottom=False)
        plt.axhline(0, c='k')
        
        fig.suptitle(
            f'$\\bf{{Control\\ Case:}}$ {self.case1}\n$\\bf{{Test\\ Case:}}$ {self.case2}\n$\\bf{{Plotting:}}$ {var}', 
            fontsize=15, horizontalalignment='left', x=0.125, y=0.98
        )
        
        plt.savefig(f'{self.path}/{var}_Figure.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def generate_html_tables(self, return_dict = None):
        cdatadef = AerosolBudgetCalculator(self.path1, self.case1, self.ss[self.ind], self.aer, reg=self.reg, loc=self.loc, mod=self.mod, land=self.land, splots=self.splot).get_tables()
        cdatase = AerosolBudgetCalculator(self.path2, self.case2, self.ss[self.ind], self.aer, reg=self.reg, loc=self.loc, mod=self.mod, land=self.land, splots=self.splot).get_tables()

        if 'year' in cdatadef.columns:
            cdatadef = cdatadef.drop('year', axis=1)
        if 'year' in cdatase.columns:
            cdatase = cdatase.drop('year', axis=1)
        
        # For mode agnostic dataframes
        for col in cdatadef.columns:
            if col not in cdatase.columns:
                cdatase[col] = np.nan

        for col in cdatase.columns:
            if col not in cdatadef.columns:
                cdatadef[col] = np.nan
        
        # Reorder columns to match
        cdatadef = cdatadef[cdatase.columns]

        cdatadiff = cdatase[cdatase.columns[1:]] - cdatadef[cdatase.columns[1:]]
        cdatarel = (cdatadiff / abs(cdatadef[cdatase.columns[1:]])) * 100

        for col in cdatarel.columns:
            df = pd.DataFrame({
                f'<a target="_blank" href="{col}_{self.case1}_{self.ss[self.ind]}_latlon_splots.png">Control Case</a>': cdatadef[col],
                f'<a target="_blank"  href="{col}_{self.case2}_{self.ss[self.ind]}_latlon_splots.png">Test Case</a>': cdatase[col],
                f'<a target="_blank" href="{col}_diff_{self.ss[self.ind]}_latlon_splots.png">Difference</a>': cdatadiff[col],
                f'<a target="_blank" href="{col}_rel_{self.ss[self.ind]}_latlon_splots.png">Relative Diff (%)</a>': cdatarel[col]
            })

            if self.ind == 0:
                unit = '(#/mg-air/yr)' if 'num' in self.aer else '(Tg/yr)'
                self.gen_budget_barPlots(df, col, unit=unit)

            pd.options.display.float_format = '{:g}'.format
            df = df.map(rounding).astype(str)
            dfhtml = get_html_table(df)
            htable = dfhtml.replace(
                '<thead>',
                f'\n<caption style="font-family: Century Gothic, sans-serif; font-size: medium; text-align: left; padding: 5px; width: auto"><strong>CNTL:</strong> {self.case1}</caption>'
                f'\n<caption style="font-family: Century Gothic, sans-serif; font-size: medium; text-align: left; padding: 5px; width: auto"><strong>TEST:</strong> {self.case2}</caption>'
                f'\n<caption style="font-family: Century Gothic, sans-serif; font-size: medium; text-align: left; padding: 5px; width: auto"><strong>VRBL:</strong> {col}</caption>\n<thead>'
            )

            with open(f'{self.path}/{col}_{self.ss[self.ind]}.html', 'w') as f:
                f.write(htable)
        rvars = rearrange_variables(list(cdatarel.columns))
        if return_dict != None:
            return_dict[self.aer] = rvars
        return cdatarel.columns
    
    def get_budget_SpData(self):
        d1 = xr.open_dataset(self.path1+self.case1+'_'+self.aer+'_'+self.ss[self.ind]+'_allVdata.nc')
        d2 = xr.open_dataset(self.path2+self.case2+'_'+self.aer+'_'+self.ss[self.ind]+'_allVdata.nc')
        diff = d2 - d1
        rel = (diff/abs(d1))*100
        vlist = list(d1.variables.keys())
        return d1, d2, diff, rel, vlist
    
    def gen_budget_spatialMaps(self,d1,d2,diff,rel,vlist,col,scrip_file):
        if col == self.aer:
            var_a1 = fnmatch.filter(vlist,'*'+col+'*+*')
        else:
            var_a1 = fnmatch.filter(vlist,'*'+col+'*')
        vname = ['Burden','Dry deposition','Wet deposition','surface emission',\
                 'elevated emission','condensation-aging','gravitational','turbulent',\
                 'incloud, stratiform','incloud, convective','belowcloud, strat.',\
                  'belowcloud, convec.','rain evap, strat.','rain evap, convec.',\
                 'renaming (sfgaex2)','coagulation (sfcoag1)','calcsize (sfcsiz3)',\
                 'calcsize (sfcsiz4)','dropmixnuc (mixnuc1)','cloudchem (AQH2SO4)',\
                 'cloudchem (AQSO4)','sfnnuc1','Aq. chem (gas-species)','gas chem/wet dep. (gas-species)']
        levels = [None,None,None,[-100,-70,-50,-20,-10,-5,-2,2,5,10,20,50,70,100]]
        exts = [col+' (Control)',col+' (Test)',col+' (diff)',col+' (rel diff %)']
        colMaps = [amwg256_map,amwg256_map,BlueWhiteOrangeRed_map,BlueWhiteOrangeRed_map]
        sei = [[5,-20,2],[5,-20,2],[0,-1,5],[0,-1,5]]
        for newVdata,t,lev,ext,cm,cbvals in zip([d1, d2, diff, rel],[self.case1,self.case2,'diff','rel'],levels,exts,colMaps,sei):
            cbs, cbe, cbi = cbvals
            i=1
            fig=plt.figure(figsize=(18,18))
            for var, name in zip(var_a1[:],vname[:]):
                print(name,newVdata[var].shape)
                ax = plt.subplot(6,4,i,projection=crs.PlateCarree())
                if ~np.isnan(newVdata[var]).all():
                    try:
                        get_plots( newVdata[var],ax=ax,cmap=cm,levels=lev,labelsize=6,\
                                     scrip_file=scrip_file,gridLines=False,\
                                        lon_range=[-180,180], lat_range=[-90,90],btm=0.03,cbthk=0.01,
                                        unit='',colbar=True,cbs=cbs,cbe=cbe,cbi=cbi).get_map()
                    except:
                        get_plots( newVdata[var],ax=ax,cmap=cm,levels=lev,labelsize=6,\
                                scrip_file='',gridLines=False,\
                                lon_range=[-180,180], lat_range=[-90,90],btm=0.03,cbthk=0.01,
                                unit='',colbar=True,cbs=cbs,cbe=cbe,cbi=cbi).get_map()
                ax.text(0.01,0.97,name,size=12,transform=ax.transAxes,va='top',bbox={'facecolor':'white','pad':1,'edgecolor':'none'})
                i+=1
                fig.suptitle(r'$\bf{CNTL:}$ '+self.case1+'\n'+\
                     r'$\bf{TEST:}$ '+self.case2+'\n'+r'$\bf{VRBL:}$ '+ext,\
                     fontsize=20,horizontalalignment='left',x=0.125,y=0.96)
            col = col.replace('total_','')
            plt.savefig(str(self.path)+'/'+col+'_'+t+'_'+self.ss[self.ind]+'_latlon_splots.png',format='png',dpi=350,bbox_inches='tight',pad_inches=0.1)
            plt.close()
    