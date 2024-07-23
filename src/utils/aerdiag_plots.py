import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
import pandas as pd

from src.utils.asediag_utils import rounding, gen_colbar_range
# Optional import
try:
    from src.scrip_utils.gen_scrip_file import gen_scrip
except ImportError:
    pass
    
class get_plots(object):
    """
    A class to generate and configure spatial 2D maps using cartopy and matplotlib.
    This should work for both regular lat-lon and spectral element grids.
    """
    
    def __init__(self,var,ax,**kwargs):
        self.var = var
        self.ax = ax
        self.xint = kwargs.get('xint',None)
        self.yint = kwargs.get('yint',None)
        self.figsize = kwargs.get('figsize',None)
        self.scrip_file = kwargs.get('scrip_file',None)
        self.lat_range = kwargs.get('lat_range',[-90,90])
        self.lon_range = kwargs.get('lon_range',[-180,180])
        self.cm = kwargs.get('cmap',plt.cm.jet)
        self.labelsize = kwargs.get('labelsize',13)
        self.unit = kwargs.get('unit','unit')
        self.gridLines = kwargs.get('gridLines',True)
        self.colbar = kwargs.get('colbar',True)
        self.map_proj = kwargs.get('projection',crs.PlateCarree())
        self.res = kwargs.get('res','110m')
        self.cbs = kwargs.get('cbs',0)
        self.cbe = kwargs.get('cbe',-1)
        self.cbi = kwargs.get('cbi',1)
        self.verts = kwargs.get('verts',None)
        self.rr = kwargs.get('levels',None)
        self.btm = kwargs.get('btm',0.06)
        self.cbthk = kwargs.get('cbthk',0.02)
    
        
    def get_verts(self):
        """
        Retrieve vertices for plotting from scrip file or process scrip metadata.
        This step is essential for plotting in spectral element grid.
        """
        try:
            corner_lon,corner_lat,center_lon,center_lat = gen_scrip(res=self.scrip_file).get_scrip_file()
        except:
            ds_scrip=xr.open_dataset(self.scrip_file)
            corner_lon = np.copy( ds_scrip.grid_corner_lon.values )
            corner_lat = np.copy( ds_scrip.grid_corner_lat.values )
            center_lon = np.copy( ds_scrip.grid_center_lon.values )
            
        if ((np.min(self.lon_range) < 0) & (np.max(corner_lon) > 180)):
            corner_lon[corner_lon > 180.] -= 360.
        
        lons_corners = np.copy(corner_lon.reshape(corner_lon.shape[0],corner_lon.shape[1],1))
        lats_corners = np.copy(corner_lat.reshape(corner_lat.shape[0],corner_lat.shape[1],1))
        lons_corners[lons_corners > 180.] -= 360
        center_lon[center_lon > 180.] -= 360
        
        lon_maxmin = np.max(lons_corners,axis=(1,2)) - np.min(lons_corners,axis=(1,2))
        g180 = np.where(lon_maxmin>180)[0]
        g180l0 = np.where(np.mean(lons_corners[g180],axis=(1,2)) <= 0)[0]
        tmp_lons_corners = lons_corners[g180[g180l0]].copy()
        tmp_lons_corners = np.where(lons_corners[g180[g180l0]]<0,180,tmp_lons_corners)
        lons_corners = np.append(lons_corners,tmp_lons_corners,axis=0)
        lats_corners = np.append(lats_corners,lats_corners[g180[g180l0]],axis=0)
        lons_corners[g180[g180l0]] = np.where(lons_corners[g180[g180l0]]>0,-180,lons_corners[g180[g180l0]])
        self.var = np.append(self.var,self.var[g180[g180l0]],axis=0)
        
        g180g0 = np.where(np.mean(lons_corners[g180],axis=(1,2)) > 0)[0]
        tmp_lons_corners = lons_corners[g180[g180g0]].copy()
        tmp_lons_corners = np.where(lons_corners[g180[g180g0]]>0,-180,tmp_lons_corners)
        lons_corners = np.append(lons_corners,tmp_lons_corners,axis=0)
        lats_corners = np.append(lats_corners,lats_corners[g180[g180g0]],axis=0)
        lons_corners[g180[g180g0]] = np.where(lons_corners[g180[g180g0]]<0,180,lons_corners[g180[g180g0]])
        self.var = np.append(self.var,self.var[g180[g180g0]],axis=0)

        verts = np.concatenate((lons_corners, lats_corners), axis=2)
            
        return self.var, verts
        
    def get_map(self):
        """
        Generate and display the map plot.
        """
        kwd_polycollection = {}
        kwd_pcolormesh = {}
        if self.gridLines == True:
            kwd_polycollection['edgecolor'] = 'k'
            kwd_polycollection['lw'] = 0.05
            kwd_pcolormesh['edgecolors'] = 'k'
            kwd_pcolormesh['lw'] = 0.01
        plt.rcParams['font.family'] = 'STIXGeneral'
        ## levels
        if self.rr == None:
            var1 = self.var.stack(grid=self.var.dims)
            var1 = var1.dropna("grid", how="all")
            self.rr = gen_colbar_range(v1=var1,v2=var1).hmap()
        ranges=self.rr
        self.ax.set_global()
        clen=len(np.arange(0,257)[self.cbs:self.cbe:self.cbi])
        try:
            self.cm = ListedColormap(self.cm.colors[self.cbs:self.cbe:self.cbi])
        except:
            self.cm = self.cm
            print('Cannot subscript Segmented Colormap!')
        if ('.nc' in str(self.scrip_file)) | (self.scrip_file.isdigit()):
            var, verts = self.get_verts()
            im = PolyCollection(verts,cmap=self.cm,**kwd_polycollection,\
                               norm=matplotlib.colors.BoundaryNorm(boundaries=ranges, ncolors=clen) )
            im.set_array(var)
            self.ax.add_collection(im)
        else:
            try:
                lon = self.var.lon
                lat = self.var.lat
            except:
                lon = self.var.longitude
                lat = self.var.latitude
            im = self.ax.pcolormesh(lon, lat, self.var, cmap=self.cm, transform=self.map_proj, \
                                    **kwd_pcolormesh, norm=matplotlib.colors.BoundaryNorm(boundaries=ranges, ncolors=clen) )
        
        self.ax.set_xlim(self.lon_range)
        if self.xint == None:
            self.xint = np.around((self.lon_range[1]-self.lon_range[0])/6)
        xticklabels=np.arange(self.lon_range[0],self.lon_range[1]+self.xint,self.xint)
        self.ax.set_ylim(self.lat_range)
        if self.yint == None:
            self.yint = np.around((self.lat_range[1]-self.lat_range[0])/6)
        yticklabels=np.arange(self.lat_range[0],self.lat_range[1]+self.yint,self.yint)
        self.ax.coastlines(resolution=self.res,lw=0.5,edgecolor='k')
        self.ax.add_feature(cfeature.BORDERS.with_scale(self.res),lw=0.5,edgecolor='k')
        self.ax.set_xticks(xticklabels,crs=self.map_proj)
        self.ax.set_yticks(yticklabels,crs=self.map_proj)
        self.ax.tick_params(labelsize=self.labelsize)
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        self.ax.xaxis.set_major_formatter(lon_formatter)
        self.ax.yaxis.set_major_formatter(lat_formatter)
        self.ax.grid( lw=0.5, color='#EBE7E0', alpha=0.5, linestyle='-.')
        ## Take care of the colorbar
        fig = self.ax.figure
        ## Dynamic page size depending on the lat/lon ranges or panel size
        if self.figsize != None:
            positions = self.ax.get_position()
            gapy = positions.y0-positions.y1
            gapx = positions.x0-positions.x1
            ratio = gapy/gapx
            if (ratio < 0.6) and (ratio > 0.4):
                self.figsize.set_size_inches(18,10,forward=True)
                plt.draw()
            elif (ratio < 0.4):
                self.figsize.set_size_inches(18,7,forward=True)
                plt.draw()
            elif (ratio > 1) and (ratio < 1.3):
                self.figsize.set_size_inches(16,16,forward=True)
                plt.draw()
            elif (ratio > 1.3):
                self.figsize.set_size_inches(14,18,forward=True)
                plt.draw()

        positions = self.ax.get_position()
        if self.colbar==True:
            if ranges is not None and \
            not all(v == 0 for v in ranges) and \
            not any(np.isnan(v) for v in ranges) and \
            not any(np.isinf(v) for v in ranges):

                ## rounding the colorbar ticks
                s1 = pd.DataFrame(ranges)
                s2 = s1.map(lambda x: rounding(x))[0].tolist()
                cbar_ticks=list(map(str,s2))
                cbar_ticks = [i.replace('.0','') if i[-2:]=='.0' else i for i in cbar_ticks]
                cbar_ticks = [i.rstrip('0') if '.' in i else i for i in cbar_ticks]
                if len(cbar_ticks) > 12:
                    cbar_ticks[::2]=['']*len(cbar_ticks[::2])
                else:
                    cbar_ticks[0]=''
                    cbar_ticks[-1]=''
                cax = fig.add_axes([positions.x0,positions.y0-self.btm,positions.x1-positions.x0,self.cbthk])
                cbar = fig.colorbar(im,cax=cax,orientation='horizontal',ticks=ranges,extend='neither',fraction=0.08,drawedges=True)
                cbar.ax.set_xticklabels(cbar_ticks, size=self.labelsize )
                cbar.set_label(label=self.unit,size=self.labelsize)
                cbar.outline.set_linewidth(1.5)
                cbar.dividers.set_linewidth(1.5)
        ## panel box thickness
        plt.setp(self.ax.spines.values(),lw=1.5)
        return im
