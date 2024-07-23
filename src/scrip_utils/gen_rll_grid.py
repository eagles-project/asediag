import numpy as np
import xarray as xr
from pathlib import Path
import logging

from src.utils.asediag_utils import get_dir_path

class gen_RLL(object):
    """
    A class for generating a regular latitude-longitude (RLL) grid.

    Args:
        res (str): Resolution of the grid in the format 'lat_resolution x lon_resolution'. Default is '180x360'.
        file (str): Input file name containing lon and lat coordinates. Default is None.
        nc (bool): Flag indicating whether to generate NetCDF output. Default is False.
        fdir (str): Input file directory. Default is an empty string.
        path (str): Output file directory. Default is an empty string.
    """
    
    def __init__(self,**kwargs):
        """
        Initializes the gen_RLL object with the provided arguments.
        """
        self.nResolution = kwargs.get('res', '180x360')
        self.file = kwargs.get('file', None)
        self.nc = kwargs.get('nc', False)
        self.fdir = kwargs.get('fdir', '')
        self.path = kwargs.get('path', '')
        self.xdim = kwargs.get('xdim', 'lon')
        self.ydim = kwargs.get('ydim', 'lat')
        self.logger = logging.getLogger(str(get_dir_path(self.path))+'/log.ggen')
        
    def get_rll(self):
        """
        Generates the regular latitude-longitude (RLL) grid.

        Returns:
            If nc is True:
                str: Path to the generated NetCDF file.
            If nc is False:
                tuple: A tuple containing the following grid information:
                    - coord (ndarray): Array of shape (3, num_nodes) representing the grid coordinates.
                    - Fixedfaces (ndarray): Array of shape (num_faces, 4) representing the faces of the grid.
                    - oldfaces (ndarray): Array of shape (num_faces, 4) representing the original faces of the grid.
                    - n_lat (int): Number of latitudes in the grid.
                    - n_lon (int): Number of longitudes in the grid.
                    - rank (int): A constant value 2 representing the number of dimensions of the grid.
        """
        if self.file != None:
            if Path(str(self.file)).is_file():
                data = xr.open_dataset(str(self.file))
            else:
                dir_path = get_dir_path(self.fdir)
                data = xr.open_dataset(dir_path / str(self.file))
            lon = data[self.xdim].values
            lat = data[self.ydim].values
            n_lon = data.dims[self.xdim]
            n_lat = data.dims[self.ydim]
            
            dlon_first = lon[1] - lon[0]
            dlon_second = lon[n_lon-1] - lon[n_lon-2]
            dlon_last = lon[0] - (lon[n_lon-1] - 360)
            
            if np.fabs(dlon_first-dlon_last) < 1e-12:
                force_global = True
            
            lon_edges = np.zeros(n_lon+1)
            
            if force_global:
                lon_edges[0] = 0.5*(lon[0] + lon[n_lon-1] - 360)
                lon_edges[n_lon] = lon_edges[0] + 360
            else:
                lon_edges[0] = lon[0] - 0.5*dlon_first
                lon_edges[n_lon] = lon[n_lon-1] + 0.5*dlon_second
                
            lon_edges[n_lon] = lon_edges[0] + 360
            lon_edges[1:-1] = 0.5*(lon[1:]+lon[:-1])
            
            lat_edges = np.zeros(n_lat+1)
            lat_edges[0] = lat[0] - 0.5*(lat[1] - lat[0])
            lat_edges[0] = np.where(lat_edges[0]<-90,-90,lat_edges[0])
            lat_edges[0] = np.where(lat_edges[0]>90,90,lat_edges[0])
            lat_edges[n_lat] = lat[n_lat-1] + 0.5*(lat[n_lat-1] - lat[n_lat-2])
            lat_edges[n_lat] = np.where(lat_edges[n_lat]>90,90,lat_edges[n_lat])
            lat_edges[n_lat] = np.where(lat_edges[n_lat]<-90,-90,lat_edges[n_lat])
            lat_edges[1:-1] = 0.5*(lat[:-1] + lat[1:])
    
            lon_edges = lon_edges*(np.pi/180.)
            lat_edges = lat_edges*(np.pi/180.)
        else:
            n_lon=int(self.nResolution.split('x')[1])
            n_lat=int(self.nResolution.split('x')[0])
            lon_edges = np.arange(0.0,361,360/n_lon)*np.pi/180
            lat_edges = np.arange(-90,91,180/n_lat)*np.pi/180

        wrap = np.fmod(lon_edges[-1]-lon_edges[0],2*np.pi) < 1e-12
        need_sp_node = abs(lat_edges[0]+0.5*np.pi) < 1e-12
        need_np_node = abs(lat_edges[-1]-0.5*np.pi) < 1e-12
        sp_offset = 1 if need_sp_node else 0
        beg = 1 if need_sp_node else 0
        end = n_lat-1 if need_np_node else n_lat
        n_nodes = n_lon if wrap else n_lon+1
        
        nodes = []
        if need_sp_node:
            nodes.append([0,0,-1])
        # Create meshgrid of lat and lon
        dlambda, dphi = np.meshgrid(lon_edges[:n_nodes], lat_edges[beg:end+1])
        dX = np.cos(dphi) * np.cos(dlambda)
        dY = np.cos(dphi) * np.sin(dlambda)
        dZ = np.sin(dphi)
        nodes.extend(np.dstack((dX, dY, dZ)).reshape(-1, 3).tolist())

        if need_np_node:
            nodes.append([0,0,1])
        coord = np.array(nodes).T

        flipped = False
        if (lat_edges[1]<lat_edges[0]):
            flipped = not flipped
        if (lon_edges[1]<lon_edges[0]):
            flipped = not flipped

        faces=[]
        if need_sp_node:
            for i in range(n_lon):
                face = []
                face.append(0)
                face.append((i+1) % n_nodes+1 if not flipped else 0)
                face.append(i+1)
                face.append(0 if not flipped else (i+1) % n_nodes+1)
                
                faces.append(face)

        jx = np.arange(beg, end) - beg
        ind = jx * n_nodes + sp_offset
        ind_next = (jx + 1) * n_nodes + sp_offset
        i = np.arange(n_lon)
        face = np.empty((4, len(jx), len(i)), dtype=int)
        face[0] = np.roll(ind[:, None] + i[None, :], -1, axis=1)
        face[1] = ind_next[:, None] + (i[None, :] + 1) % n_nodes if not flipped else ind[:, None] + i[None, :]
        face[2] = ind_next[:, None] + i[None, :]
        face[3] = ind[:, None] + i[None, :] if not flipped else ind_next[:, None] + (i[None, :] + 1) % n_nodes
        faces.extend(face.transpose(1, 2, 0).reshape(-1, 4))
        if need_np_node:
            jx = n_lat - beg -1
            ind = jx*n_nodes + sp_offset
            ind_next = (jx+1)*n_nodes + sp_offset
            np_idx = len(nodes) -1
            for i in range(n_lon):
                face = []
                face.append(np_idx)
                face.append(ind+i if not flipped else np_idx)
                face.append(ind + (i+1) % n_nodes)
                face.append(np_idx if not flipped else ind+i)
                faces.append(face) 
        oldfaces = np.array(faces,dtype=np.int32)
        Fixedfaces = np.array(faces,dtype=np.int32)+1
        if self.nc == True:
            data_vars = {
                        'coord':(['num_dim','num_nodes'], coord),
                        'connect1':(['num_el_in_blk1','num_nod_per_el1'], Fixedfaces),
                        'global_id1':(['num_el_in_blk1'],np.arange(len(Fixedfaces))),
                          }
    
            coords = {'num_el_in_blk1': (['num_el_in_blk1'], np.arange(np.array(Fixedfaces).shape[0],dtype=np.int32)),
                      'num_nod_per_el1': (['num_nod_per_el1'], np.arange(4,dtype=np.int32)),
                      'num_dim': (['num_dim'], np.arange(3)),
                      'num_nodes': (['num_nodes'], np.arange(coord.shape[1])),
                      'num_el_blk':(['num_el_blk'],np.zeros(1)),
                      }
            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            ds.attrs = {'version': np.array(5,dtype=np.float32),
                       'rectilinear': 'true',
                        'rectilinear_dim0_size': np.array(n_lat,dtype=np.int32),
                        'rectilinear_dim1_size': np.array(n_lon,dtype=np.int32),
                        'rectilinear_dim0_name': self.ydim,
                        'rectilinear_dim1_name': self.xdim}
            dir_path = get_dir_path(self.path)
            self.logger.info('\nCreating RLL grid file '+str('RLL_'+str(n_lat)+'x'+str(n_lon)+'.g')+' in ' + str(dir_path))
            ds.load().to_netcdf(dir_path / str('RLL_'+str(n_lat)+'x'+str(n_lon)+'.g'))
            return dir_path / str('RLL_'+str(n_lat)+'x'+str(n_lon)+'.g')
        else:   
            self.logger.info('\nGenerating RLL grid metadata')
            return coord, Fixedfaces, oldfaces, n_lat, n_lon, 2

