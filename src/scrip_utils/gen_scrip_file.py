import numpy as np
from pathlib import Path
import xarray as xr
import logging

from src.scrip_utils.get_face_area import calculate_face_area_quadrature_method
from src.scrip_utils.gen_rll_grid import gen_RLL
from src.scrip_utils.gen_phys_grids import gen_pg
from src.utils.asediag_utils import get_dir_path

class gen_scrip(object):
    """
    A class for generating a SCRIP file.

    Args:
        res (str): Resolution of the grid. Default is None.
        nc (bool): Flag indicating whether to generate NetCDF output. Default is False.
        path (str): Output file directory. Default is an empty string.
        file (str): Input file name. Default is None.
        fdir (str): Input grid file directory. Default is an empty string.
        grid (str): Grid file name. Default is None.
        xdim (str): Dimension name for longitude. Default is 'lon'.
        ydim (str): Dimension name for latitude. Default is 'lat'.
    """
    
    def __init__(self,**kwargs):
        """
        Initializes the gen_scrip object with the provided arguments.
        """
        self.nResolution = kwargs.get('res', None)
        self.nc = kwargs.get('nc', False)
        self.path = kwargs.get('path', '')
        self.file = kwargs.get('file', None)
        self.fdir = kwargs.get('fdir', '')
        self.grid = kwargs.get('grid', None)
        self.xdim = kwargs.get('xdim', 'lon')
        self.ydim = kwargs.get('ydim', 'lat')
        self.logger = logging.getLogger(str(get_dir_path(self.path))+'/log.ggen')
    
    @staticmethod
    def cartesian_to_latlon(dX,dY,dZ):
        """
        Converts Cartesian coordinates to latitude and longitude.

        Args:
            dX (ndarray): Array of x-coordinates.
            dY (ndarray): Array of y-coordinates.
            dZ (ndarray): Array of z-coordinates.

        Returns:
            tuple: A tuple containing the longitude and latitude arrays.
        """
        dMag = dX**2 + dY**2 + dZ**2
        dMag = np.sqrt(dMag)
        
        dX /= dMag
        dY /= dMag
        dZ /= dMag
        tol = 1e-20
        lon_deg = np.zeros(dZ.shape[0])
        lon_deg = np.where(np.fabs(dZ)< (1.0 - tol),(np.arctan2(dY,dX) * 180.0 / np.pi),0)
        lon_deg = np.where(lon_deg<0,lon_deg+360,lon_deg)
        lat_deg = np.where(np.fabs(dZ)< (1.0 - tol),(np.arcsin(dZ) * 180.0 / np.pi),-90)
        lat_deg[(np.fabs(dZ) >= (1.0 - tol)) & (dZ>0)] = 90
        return lon_deg, lat_deg
        
    def get_scrip_file(self):
        """
        Generates the SCRIP file.

        Returns:
            If nc is True:
                str: Path to the generated NetCDF file.
            If nc is False:
                tuple: A tuple containing the corner longitude, corner latitude, center longitude, and center latitude arrays.
        """ 
        if (self.file != None):
            if Path(str(self.file)).is_file():
                data = xr.open_dataset(str(self.file))
            else:
                dir_path = get_dir_path(self.fdir)
                data = xr.open_dataset(dir_path / str(self.file))
            n_lon = data.dims[self.xdim]
            n_lat = data.dims[self.ydim]
            dir_path = get_dir_path(self.path)
            fname = dir_path / str('RLL'+str(n_lat)+'x'+str(n_lon)+'_SCRIP.nc')
            if not Path(fname).is_file():
                coord, modf, faces, n_lat, n_lon, rank = gen_RLL(res=self.nResolution,fdir=self.fdir,file=self.file,xdim=self.xdim,ydim=self.ydim).get_rll()
                dims = np.array([n_lon, n_lat],dtype=np.int32)
            else:
                self.logger.info(str(fname)+' already exists!\n Using it.')
                return fname
        elif (self.grid != None):
            dir_path = get_dir_path(self.path)
            fname = dir_path / str(str(self.grid)[:-3]+'.nc')
            if not Path(fname).is_file():
                dir_path = get_dir_path(self.fdir)
                fname = dir_path / str(str(self.grid)[:-3]+'.nc')
                data = xr.open_dataset(dir_path / str(self.grid))
                coord, modf, faces = data['coord'].values, data['connect1'].values, data['connect1'].values -1
                rank = 1
                dims = np.array([1],dtype=np.int32)
            else:
                self.logger.info(str(fname)+' already exists!\n Using it.')
                return dir_path / str(str(self.grid))
        elif ('x' in self.nResolution):
            n_lon=int(self.nResolution.split('x')[1])
            n_lat=int(self.nResolution.split('x')[0])
            dir_path = get_dir_path(self.path)
            fname = dir_path / str('RLL'+str(n_lat)+'x'+str(n_lon)+'_SCRIP.nc')
            if not Path(fname).is_file():
                coord, modf, faces, n_lat, n_lon, rank = gen_RLL(res=self.nResolution,fdir=self.fdir,file=self.file,xdim=self.xdim,ydim=self.ydim).get_rll()
                dims = np.array([n_lon, n_lat],dtype=np.int32)
            else:
                self.logger.info(str(fname)+' already exists!\n Using it.')
                return fname
        else:
            dir_path = get_dir_path(self.path)
            fname = dir_path / str('ne'+str(self.nResolution)+'pg2_SCRIP.nc')
            if not Path(fname).is_file():
                coord, modf, faces, rank = gen_pg(int(self.nResolution)).get_pg2()
                dims = np.array([1],dtype=np.int32)
            else:
                self.logger.info(str(fname)+' already exists!\n Using it.')
                return fname

        nCornersMax = 4
        nElements = len(faces)
                
        cornerLon = np.empty((nElements, nCornersMax))
        cornerLat = np.empty((nElements, nCornersMax))
        centerLon = np.zeros(nElements)
        centerLat = np.zeros(nElements)
        center = np.zeros((3,nElements))
        area = calculate_face_area_quadrature_method(faces,coord).get_face_area()
        n_corners = len(faces[0])
        for j in range(n_corners):
            corner = coord[:,faces[:,j]]
            cornerLon[:,j],cornerLat[:,j] = self.cartesian_to_latlon(corner[0], corner[1], corner[2])
            center = center + corner
        center = center / n_corners
        dMag = np.sqrt(center[0]**2 + center[1]**2 + center[2]**2)

        center[0] /= dMag
        center[1] /= dMag
        center[2] /= dMag
        centerLon,centerLat = self.cartesian_to_latlon(center[0], center[1], center[2])   

        for j in range(n_corners):
            cornerLon[:,j] = np.where(cornerLat[:,j]==90,centerLon,cornerLon[:,j])
            cornerLon[:,j] = np.where(cornerLat[:,j]==-90,centerLon,cornerLon[:,j])
            londiff = centerLon - cornerLon[:,j]
            cornerLon[:,j] = np.where(londiff>180,cornerLon[:,j] + 360,cornerLon[:,j])
            cornerLon[:,j] = np.where(londiff<-180,cornerLon[:,j] - 360,cornerLon[:,j])

        if self.nc == True:
            data_vars = {'grid_area':(['grid_size'], area, {'units': 'radians^2', 'long_name':'grid_area'}),
                        'grid_center_lat':(['grid_size'], centerLat, {'units': 'degrees', 'long_name':'grid_center_lat'}),
                         'grid_center_lon':(['grid_size'], centerLon, {'units': 'degrees', 'long_name':'grid_center_lon'}),
                         'grid_corner_lat':(['grid_size','grid_corners'], cornerLat, {'units': 'degrees', 'long_name':'grid_corner_lat'}),
                         'grid_corner_lon':(['grid_size','grid_corners'], cornerLon, {'units': 'degrees', 'long_name':'grid_corner_lon'}),
                         'grid_dims':(['grid_rank'], dims, {'long_name':'grid_dims'}),
                         'grid_imask':(['grid_size'],np.ones(nElements,dtype=np.int32))
                         }
    
            coords = {'grid_size': (['grid_size'], np.arange(nElements)),
                      'grid_corners': (['grid_corners'], np.arange(nCornersMax)),
                      'grid_rank': (['grid_rank'], np.array(np.arange(rank)))
                      }
            enc = {
                    'zlib': False,
                    'shuffle': False,
                    'complevel': 0,
                    'fletcher32': False,
                    'contiguous': True,
                    'chunksizes': None,
                    'dtype': np.dtype('float64'),
                    '_FillValue': 9.96920996838687e+36}
            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            encoding = {var: enc for var in ds.data_vars if np.dtype(ds[var]) == 'float64'}
            dir_path = get_dir_path(self.path)
            if (self.grid != None):
                self.logger.info('\nCreating SCRIP file '+str(str(self.grid)[:-2]+'_SCRIP.nc')+' in ' + str(dir_path))
                ds.load().to_netcdf(dir_path / str(str(self.grid)[:-2]+'_SCRIP.nc'),encoding=encoding)
                self.logger.info('\nGenerated '+str(dir_path / str(str(self.grid)[:-2]+'_SCRIP.nc')))
                return dir_path / str(str(self.grid)[:-2]+'_SCRIP.nc')
            elif ((self.file != None) or ('x' in self.nResolution)):
                self.logger.info('\nCreating SCRIP file '+str('RLL'+str(n_lat)+'x'+str(n_lon)+'_SCRIP.nc')+' in ' + str(dir_path))
                ds.load().to_netcdf(dir_path / str('RLL'+str(n_lat)+'x'+str(n_lon)+'_SCRIP.nc'),encoding=encoding)
                self.logger.info('\nGenerated '+str(dir_path / str('RLL'+str(n_lat)+'x'+str(n_lon)+'_SCRIP.nc')))
                return dir_path / str('RLL'+str(n_lat)+'x'+str(n_lon)+'_SCRIP.nc')
            else:
                self.logger.info('\nCreating SCRIP file '+str('ne'+str(self.nResolution)+'pg2_SCRIP.nc')+' in ' + str(dir_path))
                ds.load().to_netcdf(dir_path / str('ne'+str(self.nResolution)+'pg2_SCRIP.nc'),encoding=encoding)
                self.logger.info('\nGenerated '+str(dir_path / str('ne'+str(self.nResolution)+'pg2_SCRIP.nc')))
                return dir_path / str('ne'+str(self.nResolution)+'pg2_SCRIP.nc')
        else:
            return cornerLon, cornerLat, centerLon, centerLat

