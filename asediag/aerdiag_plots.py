import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as crs
import cartopy.feature as cfeature
import cmaps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.collections import PolyCollection
import pandas as pd

def rounding(n):
    try:
        sgn = -1 if n<0 else 1
        num = format(abs(n)-int(abs(n)),'f')
        if int(num[2:])<1:
            d = (abs(n))
            return sgn * d
        else:
            for i,e in enumerate(num[2:]):
                if e!= '0':
                    if i==0:
                        d = int(abs(n)) + float(num[:i+5])
                    else:
                        d = int(abs(n)) + float(num[:i+4])
                    return sgn * d
    except:
        return np.nan
    
class get_plots(object):
    
    def __init__(self,var,ax,**kwargs):
        self.var = var
        self.ax = ax
        self.figsize = kwargs.get('figsize',None)
        self.scrip_file = kwargs.get('scrip_file',None)
        self.lat_range = kwargs.get('lat_range',[-90,90])
        self.lon_range = kwargs.get('lon_range',[-180,180])
        self.cm = kwargs.get('cmap',cmaps.amwg256)
        self.labelsize = kwargs.get('labelsize',13)
        self.unit = kwargs.get('unit','unit')
        self.gridLines = kwargs.get('gridLines',True)
        self.map_proj = kwargs.get('projection',crs.PlateCarree())
        self.rr = kwargs.get('levels',[0.,0.000274,0.00307,0.0214,0.0793,.198,.392,.682,1.13,5.,10.,32.9])
    
        
    def get_verts(self):
        ds_scrip=xr.open_dataset(self.scrip_file)
        corner_lon = np.copy( ds_scrip.grid_corner_lon.values )
        corner_lat = np.copy( ds_scrip.grid_corner_lat.values )
        center_lon = np.copy( ds_scrip.grid_center_lon.values )
        if ( (np.min(self.lon_range) < 0) & (np.max(corner_lon) > 180) ):
            corner_lon[corner_lon > 180.] -= 360.
        
        lons_corners = np.copy( corner_lon.reshape( corner_lon.shape[0],
                                                      corner_lon.shape[1],1) )
        lats_corners = np.copy( corner_lat.reshape( corner_lat.shape[0],
                                                              corner_lat.shape[1],1) )
        lons_corners[ lons_corners > 180. ] -= 360
        center_lon[ center_lon > 180. ] -= 360
        
        lons_corners_add = []
        lats_corners_add = []
        var_add = []
        # For longitudes -180, 180
        for i, cenlon in enumerate( center_lon ):
            lon_maxmin = np.max( lons_corners[i,:,:] ) - \
                         np.min( lons_corners[i,:,:] )
            if ( lon_maxmin > 180 ):
                if np.mean( lons_corners[i,:,:] ) <= 0:  
                    inds2 = np.where( lons_corners[i,:,:] < 0)[0]
                    tmp_lons_corners = np.copy( lons_corners[i,:] )
                    tmp_lons_corners[inds2] = 180.
                    lons_corners_add.append( tmp_lons_corners )
                    lats_corners_add.append( lats_corners[i,:] )
        
                    inds = np.where( lons_corners[i,:,:] > 0 )[0]
                    lons_corners[i,inds] = -180.
                    
                    var_add.append( self.var[i] )
        
                elif np.mean( lons_corners[i,:,:] ) > 0:
                    inds2 = np.where( lons_corners[i,:,:] > 0)[0]
                    tmp_lons_corners = np.copy( lons_corners[i,:] )
                    tmp_lons_corners[inds2] = -180.
                    lons_corners_add.append( tmp_lons_corners )
                    lats_corners_add.append( lats_corners[i,:] )                     
                    
                    inds = np.where( lons_corners[i,:,:] < 0 )[0]
                    lons_corners[i,inds] = 180.
        
                    var_add.append( self.var[i] )        
        lons_corners = np.concatenate( (lons_corners, np.array(lons_corners_add)), axis=0 )
        lats_corners = np.concatenate( (lats_corners, np.array(lats_corners_add)), axis=0 )
        self.var = np.concatenate( (self.var, np.array(var_add)), axis=0 )        
        verts = np.concatenate( ( lons_corners, lats_corners ), axis=2)
        return self.var, verts
        
    def get_map(self):
        var, verts = self.get_verts()
        kwd_polycollection = {}
        if self.gridLines == True:
            kwd_polycollection['edgecolor'] = 'k'
            kwd_polycollection['lw'] = 0.05
        plt.rcParams['font.family'] = 'STIXGeneral'
        ## levels
        ranges=self.rr
        self.ax.set_global()
        im = PolyCollection(verts,cmap=self.cm,**kwd_polycollection,\
                           norm=matplotlib.colors.BoundaryNorm(boundaries=ranges, ncolors=256) )
        im.set_array(var)
        self.ax.add_collection(im)
        
        self.ax.set_xlim(self.lon_range)
        xint = np.around((self.lon_range[1]-self.lon_range[0])/6)
        xticklabels=np.arange(self.lon_range[0],self.lon_range[1]+xint,xint)
        self.ax.set_ylim(self.lat_range)
        yint = np.around((self.lat_range[1]-self.lat_range[0])/6)
        yticklabels=np.arange(self.lat_range[0],self.lat_range[1]+yint,yint)
        self.ax.coastlines(resolution='110m',lw=0.5,edgecolor='k')
        self.ax.add_feature(cfeature.BORDERS.with_scale('110m'),lw=0.5,edgecolor='k')
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
        ## rounding the colorbar ticks
        s1 = pd.DataFrame(ranges)
        s2 = s1.applymap(lambda x: rounding(x) if ((abs(x)>1e-5) and (abs(x)<1e5)) else '{:.0e}'.format(x))[0].tolist()
        cbar_ticks=list(map(str,s2))
        cbar_ticks = [i.rstrip('0').rstrip('.') for i in cbar_ticks]
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
        cax = fig.add_axes([positions.x0,positions.y0-0.06,positions.x1-positions.x0,0.02])
        cbar = fig.colorbar(im,cax=cax,orientation='horizontal',ticks=ranges,extend='neither',fraction=0.08)
        cbar.ax.set_xticklabels(cbar_ticks, size=self.labelsize )
        cbar.set_label(label=self.unit,size=self.labelsize)
        ## panel box thickness
        plt.setp(self.ax.spines.values(),lw=1.5)

        