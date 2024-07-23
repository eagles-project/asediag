import numpy as np
import xarray as xr
import logging

from src.utils.asediag_utils import get_dir_path


class gen_exodus(object):
    """
    Class for generating an Exodus mesh optimized for Python
    Original C++ codebase: https://github.com/ClimateGlobalChange/tempestremap
    """
    
    def __init__(self, nResolution,**kwargs):
        """
        Initializes the gen_exodus object.

        Args:
            nResolution (int): The resolution of the mesh.
            nc (bool, optional): Indicates whether to save the mesh as a netCDF file. Defaults to False.
            path (str, optional): The path to save the mesh file. Defaults to an empty string.
        """
        self.nResolution = nResolution
        self.nc = kwargs.get('nc', False)
        self.path = kwargs.get('path', '')
        self.logger = logging.getLogger(str(get_dir_path(self.path))+'/log.ggen')
    
    @staticmethod
    def insert_cs_subnode(ix0, ix1, alpha, nodes):
        """
        Inserts a subnode between two nodes along a common edge.

        Args:
            ix0 (int): Index of the first node.
            ix1 (int): Index of the second node.
            alpha (float): Parameter determining the position of the subnode between the two nodes.
            nodes (list): List of nodes.

        Returns:
            int: Index of the inserted subnode.
        """
        x0, y0, z0 = nodes[ix0]
        x1, y1, z1 = nodes[ix1]
        dX = x0 + (x1 - x0) * alpha
        dY = y0 + (y1 - y0) * alpha
        dZ = z0 + (z1 - z0) * alpha
        dRadius = np.linalg.norm([dX, dY, dZ])
        dX /= dRadius
        dY /= dRadius
        dZ /= dRadius
        nodes.append([dX, dY, dZ])
        return len(nodes) - 1
    
    def generate_cs_multi_edge_vertices(self, ix0, ix1, nodes, edge):
        """
        Generates multiple edge vertices between two nodes.

        Args:
            ix0 (int): Index of the first node.
            ix1 (int): Index of the second node.
            nodes (list): List of nodes.
            edge (list): List to store the generated edge vertices.

        Returns:
            list: The generated edge vertices.
        """
        edge.clear()
        edge.append(ix0)
        for i in range(1, self.nResolution):
            alpha_tmp = (i / self.nResolution)
            alpha = 0.5 * (np.tan(0.25 * np.pi * (2.0 * alpha_tmp - 1.0)) + 1.0)
            ix_node = self.insert_cs_subnode(ix0, ix1, alpha, nodes)
            edge.append(ix_node)
        edge.append(ix1)
        return edge
    
    def generate_faces_from_quad(self, edge0, edge1, edge2, edge3, nodes, vecFaces):
        """
        Generates faces from a quadrilateral.

        Args:
            edge0 (list): First edge vertices.
            edge1 (list): Second edge vertices.
            edge2 (list): Third edge vertices.
            edge3 (list): Fourth edge vertices.
            nodes (list): List of nodes.
            vecFaces (list): List to store the generated faces.

        Returns:
            list: The generated faces.
        """
        edge_bot = edge0
        for j in range(self.nResolution):
            if j != self.nResolution-1:
                ix0 = edge1[j+1]
                ix1 = edge2[j+1]
                edge_top = self.generate_cs_multi_edge_vertices(ix0, ix1, nodes, [])
            else:
                edge_top = edge3
            for i in range(self.nResolution):
                vecFaces.append([edge_bot[i+1], edge_top[i+1], edge_top[i], edge_bot[i]])
            edge_bot = edge_top.copy()
        return vecFaces
    
    def gen_cs_mesh(self):
        """
        Generates the CS mesh using the specified resolution.
    
        Returns:
            tuple or None: If `nc` is True, returns None. Otherwise, returns a tuple containing the coordinate
            array, the fixed faces, and the original faces.
        """
        # Generate corner points
        dInvDeltaX = 1.0 / np.sqrt(3.0)
        nodes=[
            [dInvDeltaX, -dInvDeltaX, -dInvDeltaX],
            [dInvDeltaX, dInvDeltaX, -dInvDeltaX],
            [-dInvDeltaX, dInvDeltaX, -dInvDeltaX],
            [-dInvDeltaX, -dInvDeltaX, -dInvDeltaX],
            [dInvDeltaX, -dInvDeltaX, dInvDeltaX],
            [dInvDeltaX, dInvDeltaX, dInvDeltaX],
            [-dInvDeltaX, dInvDeltaX, dInvDeltaX],
            [-dInvDeltaX, -dInvDeltaX, dInvDeltaX]
        ]

        vecMultiEdges = [[]]*12
        for i,j,k in zip([0,1,2,3,0,1,2,3,4,5,6,7],[1,2,3,0,4,5,6,7,5,6,7,4],range(12)):
            vecMultiEdges[k] = self.generate_cs_multi_edge_vertices(i, j, nodes,[])
        faces = []
        faces.extend(self.generate_faces_from_quad(vecMultiEdges[0], vecMultiEdges[4], vecMultiEdges[5], vecMultiEdges[8], nodes,[]))
        faces.extend(self.generate_faces_from_quad(vecMultiEdges[1], vecMultiEdges[5], vecMultiEdges[6], vecMultiEdges[9], nodes,[]))
        faces.extend(self.generate_faces_from_quad(vecMultiEdges[2], vecMultiEdges[6], vecMultiEdges[7], vecMultiEdges[10], nodes,[]))
        faces.extend(self.generate_faces_from_quad(vecMultiEdges[3], vecMultiEdges[7], vecMultiEdges[4], vecMultiEdges[11], nodes,[]))
        faces.extend(self.generate_faces_from_quad(vecMultiEdges[2][::-1], vecMultiEdges[3], vecMultiEdges[1][::-1], vecMultiEdges[0], nodes,[]))
        faces.extend(self.generate_faces_from_quad(vecMultiEdges[8], vecMultiEdges[11][::-1], vecMultiEdges[9], vecMultiEdges[10][::-1], nodes,[]))

        Fixedfaces = (np.array(faces)+1).tolist()
        for i in range(len(Fixedfaces)):
            Fixedfaces[i] = Fixedfaces[i][-1:]+Fixedfaces[i][:3]
            faces[i] = faces[i][-1:]+faces[i][:3]
        
        transposed_list = [list(i) for i in zip(*nodes)]
        coord = np.array(transposed_list)
        
        if self.nc == True:
            data_vars = {
                'coord':(['num_dim','num_nodes'],coord),
                'connect1':(['num_el_in_blk1','num_nod_per_el1'],Fixedfaces),
                }
            coords = {
                'num_el_in_blk1':(['num_el_in_blk1'],np.arange(np.array(faces).shape[0])),
                'num_nod_per_el1':(['num_nod_per_el1'],np.arange(4)),
                'num_dim':(['num_dim'],np.arange(3)),
                'num_nodes':(['num_nodes'],np.arange(coord.shape[1]))
                }
            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            dir_path = get_dir_path(self.path)
            self.logger.info('\nCreating exodus file '+str('ne'+str(self.nResolution)+'.g')+' in ' + str(dir_path))
            ds.load().to_netcdf(dir_path / str('ne'+str(self.nResolution)+'.g'))
        else:
            self.logger.info('\nGenerating exodus metadata')
            return coord, Fixedfaces, faces
        
