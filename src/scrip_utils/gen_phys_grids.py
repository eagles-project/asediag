import logging
import numpy as np
import pandas as pd
import xarray as xr

from src.scrip_utils.gen_exodus import gen_exodus
from src.utils.asediag_utils import get_dir_path, group_duplicate_index

class gen_pg(object):
    """
    Generates a pg2 metadata or grid file based on the C++ code from the tempestremap project.

    Args:
        nResolution (int): The resolution parameter.
        nc (bool, optional): Flag indicating whether to generate a grid file in NetCDF format. Defaults to False.
        path (str, optional): The path to save the grid file. Defaults to ''.
    """
    
    def __init__(self, nResolution,**kwargs):
        self.nResolution = nResolution
        self.nc = kwargs.get('nc', False)
        self.path = kwargs.get('path', '')
        self.logger = logging.getLogger(str(get_dir_path(self.path))+'/log.ggen')
    
    @staticmethod
    def get_faces(tmp):
        """
        Get the face values from a DataFrame.

        Args:
            tmp (pd.DataFrame): Input DataFrame containing face values.

        Returns:
            np.ndarray: Array of face values.
        """
        fvals = np.zeros(len(tmp))
        cc=tmp[tmp.duplicated()].index.values
        ff=tmp[~tmp.duplicated()].index.values
        dd=group_duplicate_index(tmp)
        dd_df = pd.DataFrame(sorted(dd, key=lambda x: x[0])).reset_index(drop=True)
        nnans = len(dd_df[3]) - len(dd_df[3].unique())
        vals=-1*np.arange(1,nnans+2)
        uvals = pd.DataFrame(vals,columns=[3])
        ninds = dd_df[np.isnan(dd_df[3])].index.values
        uvals.index = ninds
        dd_dfNew = dd_df.fillna(uvals)

        c1v = pd.Index(dd_dfNew[1]).get_indexer(cc)
        c1vnan = np.where(c1v<0,0,c1v)
        c2v = pd.Index(dd_dfNew[2]).get_indexer(cc)
        c2vnan = np.where(c2v<0,0,c2v)
        c3v = pd.Index(dd_dfNew[3]).get_indexer(cc)
        c3vnan = np.where(c3v<0,0,c3v)
        cv = c1vnan + c2vnan + c3vnan
        cv2 = pd.Index(dd_dfNew[0]).get_indexer(ff)
        fvals[ff]=cv2
        fvals[cc]=cv
        fvals = fvals.reshape(int(len(tmp)/4),4).astype(int)
        return fvals
    
    @staticmethod
    def forward_mapping(node0, node1, node2, node3, dA, dB):
        """
        Perform forward mapping based on node coordinates and interpolation factors.

        Args:
            node0 (np.ndarray): Node coordinates for the first node.
            node1 (np.ndarray): Node coordinates for the second node.
            node2 (np.ndarray): Node coordinates for the third node.
            node3 (np.ndarray): Node coordinates for the fourth node.
            dA (float): Interpolation factor for the first dimension.
            dB (float): Interpolation factor for the second dimension.

        Returns:
            np.ndarray: Mapped node coordinates.
        """
        nodeRef = (1.0 - dA) * (1.0 - dB) * node0 + dA * (1.0 - dB) * node1 + dA * dB * node2 + (1.0 - dA) * dB * node3
        dMag = np.sqrt(np.sum(nodeRef ** 2, axis=0))
        nodeRef = nodeRef / dMag[np.newaxis,:]
        return nodeRef
    
    def get_pg2(self):
        """
        Generates the pg2 metadata or grid file.

        Returns:
            tuple: A tuple containing the coordinate array, fixed faces, old faces, and a flag indicating success.
        """
        nodes, ff, faces = gen_exodus(self.nResolution).gen_cs_mesh()
        faces = np.array(faces)
        nElements = len(faces)
        nP = 2
        dG = np.array((0.0,1.0))
        dW = np.array((0.5,0.5))
        dAccumW = np.zeros(nP+1)
        dAccumW[1:]=np.cumsum(dW)
        
        node0 = nodes[:, faces[:, 0]]
        node1 = nodes[:, faces[:, 1]]
        node2 = nodes[:, faces[:, 2]]
        node3 = nodes[:, faces[:, 3]]
        pxvals = np.array([0, 1, 1, 0, 1, 2, 2, 1, 0, 1, 1, 0, 1, 2, 2, 1])
        qxvals = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2])

        bb=np.zeros((3,nElements*16))
        i=0
        for px,qx in zip(pxvals[:],qxvals[:]):
            node_out = self.forward_mapping(node0, node1, node2, node3, np.array(dAccumW[px]), np.array(dAccumW[qx]))
            node_out = node_out.T
            bb.T[i::16]=node_out
            i=i+1
        tmp = pd.DataFrame(bb.T)
        newNodes = tmp.drop_duplicates().to_numpy()
        fvals = self.get_faces(tmp)
        coord = newNodes.T
        Fixedfaces = fvals+1
        oldfaces = fvals
        
        if self.nc == True:
            data_vars = {
                'coord':(['num_dim','num_nodes'],coord),
                'connect1':(['num_el_in_blk1','num_nod_per_el1'],Fixedfaces),
                }
            coords = {
                'num_el_in_blk1':(['num_el_in_blk1'],np.arange(np.array(oldfaces).shape[0])),
                'num_nod_per_el1':(['num_nod_per_el1'],np.arange(4)),
                'num_dim':(['num_dim'],np.arange(3)),
                'num_nodes':(['num_nodes'],np.arange(coord.shape[1]))
                }
            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            dir_path = get_dir_path(self.path)
            self.logger.info('\nCreating pg2 grid file '+str('ne'+str(self.nResolution)+'pg2.g')+' in ' + str(dir_path))
            ds.load().to_netcdf(dir_path / str('ne'+str(self.nResolution)+'pg2.g'))
        else:  
            self.logger.info('\nGenerating pg2 metadata')
            return coord, Fixedfaces, oldfaces, 1

