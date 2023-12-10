""""
    Implemetation for lab 5 session
    author: César Borja Moreno and Nerea Gallego Sánchez
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy as sc
import scipy.optimize as scOptim

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

def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)

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
    return np.array(u_vectors).squeeze()

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

        # print(points)
        points_3d.append(points/points[3])

    return np.array(points_3d)

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x

def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]],
    [x[2], 0, -x[0]],
    [-x[1], x[0], 0]], dtype="object")
    return M

def resBundleProjection_n_cameras(Op, xData, nPoints, K1, K2, D1, D2, T_w_c1, T_w_c2):
    """
    -input:
    Op: Optimization parameters: this must include a
    paramtrization for T_21 (reference 1 seen from reference 2)
    in a proper way and for X1 (3D points in ref 1)
    x1Data: (3xnPoints) 2D points on image 1 (homogeneous
    coordinates)
    x2Data: (3xnPoints) 2D points on image 2 (homogeneous
    coordinates)
    K_c: (3x3) Intrinsic calibration matrix
    nPoints: Number of points
    -output:
    res: residuals from the error between the 2D matched points
    and the projected points from the 3D points
    (2 equations/residuals per 2D point)

    ASSUMING AT LEAST 3 CAMERAS !!!
    """

    '''
    Op[0:3] -> tx, ty, tz (camera 2 in advance)
    Op[3:6] -> Rx,Ry,Rz
    ...
    Op[] -> 3DXx,3DXy,3DXz (respect to camera 1B)
    '''
    # Bundle adjustment using least squares function
    R_wa_wb = sc.linalg.expm(crossMatrix(Op[3:6]))
    t_wa_wb = Op[0:3].reshape(-1, 1)
    T_wa_wb = np.hstack((R_wa_wb, t_wa_wb))
    T_wa_wb = np.vstack((T_wa_wb, np.array([0, 0, 0, 1])))
    

    X_3D = np.hstack((Op[6:].reshape(-1, 3), np.ones((nPoints, 1))))
    T_c1a_c1b = np.linalg.inv(T_w_c1) @ T_wa_wb @ T_w_c1
    T_c1_c2 = np.linalg.inv(T_w_c1) @ T_w_c2

    x1_p = kannalaBrandtProjection(K1, D1, (T_c1a_c1b @ X_3D.T).T)
    x2_p = kannalaBrandtProjection(K2, D2, (np.linalg.inv(T_c1_c2) @ T_c1a_c1b @ X_3D.T).T)
    x3_p = kannalaBrandtProjection(K1, D1, X_3D)
    x4_p = kannalaBrandtProjection(K2, D2, (np.linalg.inv(T_c1_c2)@X_3D.T).T)

    x1 = xData[0:3, :] / xData[2, :]
    x2 = xData[3:6, :] / xData[5, :]
    x3 = xData[6:9, :] / xData[8, :]
    x4 = xData[9:12, :] / xData[11, :]

    res = np.concatenate((x1[0:2,:].T - x1_p[:,0:2], x2[0:2,:].T - x2_p[:,0:2], x3[0:2,:].T - x3_p[:,0:2], x4[0:2,:].T - x4_p[:,0:2]), axis=0)
    return res.flatten()
    

    

    


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
    T_l_r = np.loadtxt('T_leftRight.txt')

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
    # print('u1 = ', u1)

    u2 = kannalaBrandtProjection(Kc_2, d2, X)
    # print('u2 = ', u2)

    u = np.array([[503.387,450.1594,1], [267.9465,580.4671,1],[441.0609,493.0671,1]])
    X1 = kannalaBrandtUnprojection(Kc_1, d1, u)
    # print('X1 = ', X1)

    X2 = kannalaBrandtUnprojection(Kc_2, d2, u)
    # print('X2 = ', X2)

    # 2.2
    points_3d = triangulation(x1, x2, T_w_c1, T_w_c2, Kc_1, Kc_2, d1, d2, T_l_r)
    # print('points_3d = ', points_3d)

    print(points_3d.shape)
    points_3d_1 = (T_l_r @ points_3d.T).T
    x1_p = kannalaBrandtProjection(Kc_1, d1, points_3d_1)
    x2_p = kannalaBrandtProjection(Kc_2, d2, points_3d)
    print(x1_p.shape)
    print(x2_p.shape)


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img1)
    ax[0].set_title('Reprojection 3D points Camera1 A')
    plotResidual(x1.T, x1_p, 'k-', ax[0])    

    ax[1].imshow(img2)
    ax[1].set_title('Reprojection 3D points Camera2 A')
    plotResidual(x2.T, x2_p, 'k-', ax[1])
    plt.show()
    
    T_c1a_c1b = np.linalg.inv(T_w_c1) @ T_wa_wb_gt @ T_w_c1
    points_3d = triangulation(x1, x3, T_w_c1, T_w_c2, Kc_1, Kc_2, d1, d2, T_c1a_c1b)

    points_3d_1 = (T_c1a_c1b @ points_3d.T).T
    x1_p = kannalaBrandtProjection(Kc_1, d1, points_3d_1)
    x3_p = kannalaBrandtProjection(Kc_2, d2, points_3d)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img1)
    ax[0].set_title('Reprojection 3D points Camera1 A')
    plotResidual(x1.T, x1_p, 'k-', ax[0])

    ax[1].imshow(img3)
    ax[1].set_title('Reprojection 3D points Camera1 B')
    plotResidual(x3.T, x3_p, 'k-', ax[1])
    plt.show()

    # PART 3
    print('PART 3')

    T_c1a_c1b_seed = np.linalg.inv(T_w_c1) @ T_wa_wb_seed @ T_w_c1

    points_3d = triangulation(x1, x3, T_w_c1, T_w_c2, Kc_1, Kc_2, d1, d2, T_c1a_c1b_seed)
    print(points_3d.shape)
    R_wa_wb_seed = T_wa_wb_seed[0:3, 0:3]
    t_wa_wb_seed = T_wa_wb_seed[0:3, 3]
    

    Op = np.hstack((t_wa_wb_seed, crossMatrixInv(sc.linalg.logm(R_wa_wb_seed)), points_3d[:,0:3].flatten()))
    # print(Op)

    OpOptim = scOptim.least_squares(resBundleProjection_n_cameras, Op, args=(np.concatenate((x1,x2, x3,x4), axis=0), x1.shape[1], Kc_1, Kc_2, d1, d2, T_w_c1, T_w_c2), method='lm', verbose=2)

    # Print 3D model
    R_wa_wb_op = sc.linalg.expm(crossMatrix(OpOptim.x[3:6]))
    t_wa_wb_op = OpOptim.x[0:3].reshape(-1, 1)
    T_wa_wb_op = np.hstack((R_wa_wb_op, t_wa_wb_op))
    T_wa_wb_op = np.vstack((T_wa_wb_op, np.array([0, 0, 0, 1])))
    
    points_3d_op = OpOptim.x[6:].reshape(-1, 3)
    points_3d_op = np.hstack((points_3d_op, np.ones((points_3d_op.shape[0], 1))))

    points_3d_op = T_wa_wb_op @ T_w_c1 @ points_3d_op.T

    fig3D = plt.figure(7)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4,4), '-', 'W')
    drawRefSystem(ax, T_wa_wb_gt @ T_w_c1, '-', 'C1_B')
    drawRefSystem(ax, T_wa_wb_gt @ T_w_c2, '-', 'C2_B')
    drawRefSystem(ax, T_w_c2, '-', 'C2_A')

    drawRefSystem(ax, T_wa_wb_op @ T_w_c1, '-', 'C1_B_Op')
    drawRefSystem(ax, T_wa_wb_op @ T_w_c2, '-', 'C2_B_Op')

    ax.scatter(points_3d_op[0, :], points_3d_op[1, :], points_3d_op[2, :], c='r', marker='x', s=10)

    plt.title('3D model Bundle Adjustment')
    plt.show()



