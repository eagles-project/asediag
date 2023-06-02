import numpy as np

class calculate_face_area_quadrature_method(object):
    """
    Calculate the face area using the quadrature method.
    """
    
    def __init__(self, faces, nodes):
        """
        Initializes the calculate_face_area_quadrature_method object.

        Args:
            faces (ndarray): Array of face indices.
            nodes (ndarray): Array of node coordinates.
        """
        self.faces = faces
        self.nodes = nodes
    
    @staticmethod
    def refGetPoints(nCount,dG,dW):
        """
        Static method to obtain reference points and weights for the quadrature method.

        Args:
            nCount (int): Number of points.
            dG (ndarray): Array to store the reference points.
            dW (ndarray): Array to store the weights.

        Returns:
            dG (ndarray): Array of reference points.
            dW (ndarray): Array of weights.
        """
        # Check for valid range
        if nCount < 2:
            raise Exception(f"Invalid count ({nCount}): Minimum count 2")

        # Degree 2
        if nCount == 2:
            dG[0] = -1.0
            dG[1] = +1.0

            dW[0] = +1.0
            dW[1] = +1.0

        # Degree 3
        elif nCount == 3:
            dG[0] = -1.0
            dG[1] =  0.0
            dG[2] = +1.0

            dW[0] = +0.333333333333333
            dW[1] = +1.333333333333334
            dW[2] = +0.333333333333333
        # Degree 6
        elif nCount == 6:
            dG[0] = -0.9324695142031521
            dG[1] = -0.6612093864662645
            dG[2] = -0.2386191860831969
            dG[3] = +0.2386191860831969
            dG[4] = +0.6612093864662645
            dG[5] = +0.9324695142031521

            dW[0] = 0.1713244923791704
            dW[1] = 0.3607615730481386
            dW[2] = 0.4679139345726910
            dW[3] = 0.4679139345726910
            dW[4] = 0.3607615730481386
            dW[5] = 0.1713244923791704
        return dG, dW

    
    @staticmethod
    def get_points(self,n_count, d_xi0, d_xi1, d_g, d_w):
        """
        Obtain the points and weights for quadrature integration.

        Args:
            self: The object itself.
            n_count (int): Number of points.
            d_xi0 (float): Lower limit of the integration range.
            d_xi1 (float): Upper limit of the integration range.
            d_g (ndarray): Array to store the reference points.
            d_w (ndarray): Array to store the weights.

        Returns:
            d_g (ndarray): Array of reference points.
            d_w (ndarray): Array of weights.
        """
        d_g,d_w=self.refGetPoints(n_count, d_g, d_w)
        for i in range(n_count):
            d_g[i] = d_xi0 + 0.5 * (d_xi1 - d_xi0) * (d_g[i] + 1.0)
            d_w[i] = 0.5 * (d_xi1 - d_xi0) * d_w[i]
        return d_g, d_w
    
    @staticmethod
    def get_triangle_jacobian(node1, node2, node3, dA, dB):
        """
        Calculate the Jacobian of a triangle defined by three nodes.

        Args:
            node1 (ndarray): Coordinates of the first node.
            node2 (ndarray): Coordinates of the second node.
            node3 (ndarray): Coordinates of the third node.
            dA (float): Parameter A within the triangle.
            dB (float): Parameter B within the triangle.

        Returns:
            dJacobian (ndarray): Array of Jacobian values for each triangle.
        """
        dF = (1.0 - dB) * ((1.0 - dA) * node1 + dA * node2) + dB * node3
        dDaF = (1-dB)*(node2-node1)
        dDbF = -1*(1-dA)*node1-dA*node2+node3
        dInvR = 1.0 / np.sqrt(dF[0,:]**2+dF[1,:]**2+dF[2,:]**2)
        
        dDaG = [dDaF[0,:] * (dF[1,:] * dF[1,:] + dF[2,:] * dF[2,:])-dF[0,:] * (dDaF[1,:] * dF[1,:] + dDaF[2,:] * dF[2,:]),
                dDaF[1,:] * (dF[0,:] * dF[0,:] + dF[2,:] * dF[2,:])-dF[1,:] * (dDaF[0,:] * dF[0,:] + dDaF[2,:] * dF[2,:]),
                dDaF[2,:] * (dF[0,:] * dF[0,:] + dF[1,:] * dF[1,:])-dF[2,:] * (dDaF[0,:] * dF[0,:] + dDaF[1,:] * dF[1,:])]

                              
        dDbG =  [dDbF[0,:] * (dF[1,:] * dF[1,:] + dF[2,:] * dF[2,:])-dF[0,:] * (dDbF[1,:] * dF[1,:] + dDbF[2,:] * dF[2,:]),
                dDbF[1,:] * (dF[0,:] * dF[0,:] + dF[2,:] * dF[2,:])-dF[1,:] * (dDbF[0,:] * dF[0,:] + dDbF[2,:] * dF[2,:]),
                dDbF[2,:] * (dF[0,:] * dF[0,:] + dF[1,:] * dF[1,:])-dF[2,:] * (dDbF[0,:] * dF[0,:] + dDbF[1,:] * dF[1,:])]
                              
        dDenomTerm = dInvR * dInvR * dInvR
        
        dDaG[0] *= dDenomTerm
        dDaG[1] *= dDenomTerm
        dDaG[2] *= dDenomTerm

        dDbG[0] *= dDenomTerm
        dDbG[1] *= dDenomTerm
        dDbG[2] *= dDenomTerm
        
        nodeCross = np.zeros((3,dDaG[0].shape[0]))
        nodeCross[0] = dDaG[1] * dDbG[2] - dDaG[2] * dDbG[1]
        nodeCross[1] = dDaG[2] * dDbG[0] - dDaG[0] * dDbG[2]
        nodeCross[2] = dDaG[0] * dDbG[1] - dDaG[1] * dDbG[0]
        
        dJacobian = np.sqrt(nodeCross[0]**2 + nodeCross[1]**2 + nodeCross[2]**2)
        
        return dJacobian
    
    def get_face_area(self):
        """
        Calculate the area of each face using the quadrature method.

        Returns:
            d_face_area (ndarray): Array of face areas.
        """
        n_triangles = len(self.faces[0]) - 2
        n_order = 6
        dG = np.zeros(n_order)
        dW = np.zeros(n_order)
        dG, dW = self.get_points(self,n_order,0,1,dG,dW)
        d_face_area = np.zeros(self.faces.shape[0])
        for j in range(n_triangles):
            node1 = self.nodes[:, self.faces[:, 0]]
            node2 = self.nodes[:, self.faces[:, j + 1]]
            node3 = self.nodes[:, self.faces[:, j + 2]]
            for p in range(len(dW)):
                for q in range(len(dW)):
                    d_a = dG[p]
                    d_b = dG[q]
                    d_jacobian = self.get_triangle_jacobian(node1, node2, node3, d_a, d_b)
                    d_face_area += dW[p] * dW[q] * d_jacobian
        return d_face_area

