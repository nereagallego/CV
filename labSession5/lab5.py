""""
    Implemetation for lab 5 session
    author: César Borja Moreno and Nerea Gallego Sánchez
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)


def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

def plotResidual(x,xProjected,strStyle, ax):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    # Plot the line between each point and its projection also plot the point with a blue dot and the projection with a red cross
    for k in range(x.shape[0]):
        ax.plot([x[k, 0], xProjected[k, 0]], [x[k, 1], xProjected[k, 1]], strStyle)
        ax.plot(x[k, 0], x[k, 1], 'bo')
        ax.plot(xProjected[k, 0], xProjected[k, 1], 'rx')

def kannalaBrandtProjection(K, D, X):
    """
        Project a set of 3D points using the Kannala-Brandt projection model.
         -input:
             K: Camera intrinsic matrix.
             D: Distortion coefficients.
             X: 3D points.
         -output:
             x: Projected points.
         """
    u_vectors = []
    for point in X:
        r = np.sqrt(point[0]**2 + point[1]**2)
        theta = np.arctan2(r, point[2])
        phi = np.arctan2(point[1], point[0])
        d = theta + D[0] * theta**3 + D[1] * theta**5 + D[2] * theta**7 + D[3] * theta**9
        u = K @ np.array([[d * np.cos(phi)], [d * np.sin(phi)], [1]])  
        u_vectors.append(u)
    return np.array(u_vectors)

def kannalaBrandtUnprojection(K, D, u):
    """
        Unproject a set of 2D points using the Kannala-Brandt projection model.
         -input:
             K: Camera intrinsic matrix.
             D: Distortion coefficients.
             u: 2D points.
         -output:
             X: Unprojected points.
         """
    X = []
    K_inv = np.linalg.inv(K)
    for point in u:
        xc = K_inv @ point
        d = np.sqrt((xc[0]**2+xc[1]**2)/xc[2]**2)
        phi = np.arctan2(xc[1], xc[0])
        roots = np.roots([D[3], 0, D[2], 0, D[1], 0, D[0], 0, 1, -d])
        theta = 0
        for root in roots:
            if np.isreal(root):
                theta = np.real(root)
                break
        v = np.array([[np.sin(theta)*np.cos(phi)], [np.sin(theta)*np.sin(phi)], [np.cos(theta)]], dtype=np.float32)
        X.append(v)
    return np.array(X)

def triangulation(x1, x2, T_w_c1, T_w_c2, K1, K2, D1, D2, T_c1_c2):
    """
        Triangulate a set of 2D points using the Kannala-Brandt projection model.
         -input:
             x1: 2D points in the first image.
             x2: 2D points in the second image.
             T_w_c1: Pose of the first camera.
             T_w_c2: Pose of the second camera.
             K1: Intrinsic matrix of the first camera.
             K2: Intrinsic matrix of the second camera.
             D1: Distortion coefficients of the first camera.
             D2: Distortion coefficients of the second camera.
         -output:
             X: Triangulated points.
         """
    X = []
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)

    v1 = kannalaBrandtUnprojection(K1, D1, x1.T)
    v2 = kannalaBrandtUnprojection(K2, D2, x2.T)

    points_3d = []
    v1 = v1.squeeze()
    v2 = v2.squeeze()
    print(v1.shape)
    for i in range(v1.shape[0]):
        p_s1 = np.array([[-v1[i,1]], [v1[i,0]], [0], [0]], dtype=np.float32)
        p_s1 = p_s1 / np.linalg.norm(p_s1)
        p_p1 = np.array([[-v1[i,2]*v1[i,0]], [-v1[i,2]*v1[i,1]], [v1[i,0]**2+v1[i,1]**2], [0]], dtype=np.float32)
        p_p1 = p_p1 / np.linalg.norm(p_p1)
        p_s2 = np.array([[-v2[i,1]], [v2[i,0]], [0], [0]], dtype=np.float32)
        p_s2 = p_s2 / np.linalg.norm(p_s2)
        p_p2 = np.array([[-v2[i,2]*v2[i,0]], [-v2[i,2]*v2[i,1]], [v2[i,0]**2+v2[i,1]**2], [0]], dtype=np.float32)
        p_p2 = p_p2 / np.linalg.norm(p_p2)

        p_s1_2 = T_c1_c2.T @ p_s1
        p_p1_2 = T_c1_c2.T @ p_p1



        A = np.array([p_s1_2.T,p_p1_2.T,p_s2.T,p_p2.T], dtype=np.float32)
        A = A.squeeze()

        _, _, V = np.linalg.svd(A, full_matrices=True)

        points = V.T[:, -1]

        print(points)
        points_3d.append(points/points[3])

    return np.array(points_3d)


if __name__ == '__main__':
    Kc_1 = np.loadtxt('K_1.txt')
    Kc_2 = np.loadtxt('K_2.txt')

    d1 = np.loadtxt('D1_k_array.txt')
    d2 = np.loadtxt('D2_k_array.txt')

    # Pose A
    x1 = np.loadtxt('x1.txt')
    x2 = np.loadtxt('x2.txt')
    # Pose B
    x3 = np.loadtxt('x3.txt')
    x4 = np.loadtxt('x4.txt')

    T_wa_wb_gt = np.loadtxt('T_wawb_gt.txt')
    T_wa_wb_seed = np.loadtxt('T_wawb_seed.txt')
    T_w_c1 = np.loadtxt('T_wc1.txt')
    T_w_c2 = np.loadtxt('T_wc2.txt')

    img1 = cv2.imread('fisheye1_frameA.png')
    img2 = cv2.imread('fisheye2_frameA.png')
    img3 = cv2.imread('fisheye1_frameB.png')
    img4 = cv2.imread('fisheye2_frameB.png')

    # PART 2
    print('PART 2')

    # 2.1

    #  Implement the Kannala-Brandt projection and unprojection model.

    X = np.array([[3,2,10,1], [-5,6,7,1],[1,5,14,1]])
    u1 = kannalaBrandtProjection(Kc_1, d1, X)
    print('u1 = ', u1)

    u2 = kannalaBrandtProjection(Kc_2, d2, X)
    print('u2 = ', u2)

    u = np.array([[503.387,450.1594,1], [267.9465,580.4671,1],[441.0609,493.0671,1]])
    X1 = kannalaBrandtUnprojection(Kc_1, d1, u)
    print('X1 = ', X1)

    X2 = kannalaBrandtUnprojection(Kc_2, d2, u)
    print('X2 = ', X2)

    # 2.2
    points_3d = triangulation(x1, x2, T_w_c1, T_w_c2, Kc_1, Kc_2, d1, d2, T_wa_wb_gt)
    # print('points_3d = ', points_3d)

    print(points_3d.shape)
    # x1_p = kannalaBrandtProjection(Kc_1, d1, points_3d)

