""""
    Implemetation for lab 4 session
    author: César Borja Moreno and Nerea Gallego Sánchez
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.optimize as scOptim
import cv2

def resLineFitting(Op,xData):
    """
        Residual function for least squares method
        -input:
        Op: vector containing the model parameters to optimize (in this case the
        line). This description should be minimal
        xData: the measurements of our model whose residual we want to calculate
        -output:
        res: vector of residuals to compute the loss
    """
    theta = Op[0]
    d = Op[1]
    l_model = np.vstack((np.cos(theta),np.sin(theta), -d))
    res = (l_model.T @ xData).squeeze() # Since the norm is unitary the distance is easier
    res = res.flatten()
    return res

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

def plotResidual(x,xProjected,strStyle):
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
        plt.plot([x[k, 0], xProjected[k, 0]], [x[k, 1], xProjected[k, 1]], strStyle)
        plt.plot(x[k, 0], x[k, 1], 'bo')
        plt.plot(xProjected[k, 0], xProjected[k, 1], 'rx')
    


# Triangulate a set of points given the projection matrices of two cameras.
def triangulation(P1, P2, points1, points2):    
    points3D = np.zeros((4, points1.shape[1]))
    for i in range(points1.shape[1]):
        p1 = points1[:, i].reshape(2, 1)
        p2 = points2[:, i].reshape(2, 1)
        A = [p1[0] * P1[2, :] - P1[0, :], p1[1] * P1[2, :] - P1[1, :], p2[0] * P2[2, :] - P2[0, :], p2[1] * P2[2, :] - P2[1, :]]
        _, _, V = np.linalg.svd(A)
        X = V[-1, :]
        points3D[:, i] = X / X[3]

    return points3D

def points_in_front_of_both_cameras(x1, x2, T, K):
    I = np.array([[1, 0 , 0, 0], [0, 1, 0, 0], [0, 0, 1 ,0]])

    P1 = K @ I
    P2 = K @ T

    points3d = triangulation(P1, P2, x1, x2)
    points3d = points3d.T
    print(points3d.shape)

    in_front = 0

    for point in points3d:
        if point[2] < 0:
            continue

        # z > 0 in C1 frame
        pointFrame = T @ point.reshape(-1,1)
        if pointFrame[2] > 0:
            in_front += 1
    
    return in_front


# Decompose the Essential matrix
def decompose_essential_matrix(x1, x2, E, K):
    # Compute the SVD of the essential matrix
    U, _, V = np.linalg.svd(E)
    t = U[:,2].reshape(-1,1)
    
    # Ensure that the determinant of U and Vt is positive (to ensure proper rotation)

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    R_90 = U @ W @ V if np.linalg.det(U @ W @ V) > 0 else -U @ W.T @ V
    R_n90 = U @ W.T @ V if np.linalg.det(U @ W.T @ V) > 0 else -U @ W.T @ V

    # Compute the four possible solutions
    solutions = []
    solutions.append(np.hstack((R_90,U[:,2].reshape(-1,1)))) #R + 90 + t
    solutions.append(np.hstack((R_90,-U[:,2].reshape(-1,1)))) # R + 90 - t
    solutions.append(np.hstack((R_n90,U[:,2].reshape(-1,1))))  # R - 90 + t
    solutions.append(np.hstack((R_90,-U[:,2].reshape(-1,1)))) # R - 90 - t
    

    points_front = [ points_in_front_of_both_cameras(x1, x2, T, K) for T in solutions]

    T = solutions[np.argmax(points_front)]
    return T

def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    return E

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x

def crossMatrix(x):
    M = np.array([[0, -x[2], x[1]],
    [x[2], 0, -x[0]],
    [-x[1], x[0], 0]], dtype="object")
    return M


def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
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
    """

    '''
    Op[0:1] -> theta, phi
    Op[2:5] -> Rx,Ry,Rz
    Op[5:5 + nPoints*3] -> 3DXx,3DXy,3DXz
    '''
    # Bundle adjustment using least squares function
    idem = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
    R = sc.linalg.expm(crossMatrix(Op[2:5]))
    t = np.array([np.sin(Op[0])*np.cos(Op[1]), np.sin(Op[0])*np.sin(Op[1]), np.cos(Op[0])]).reshape(-1,1)
    theta_ext_1 = K_c @ idem
    T = np.hstack((R, t))
    theta_ext_2 =  K_c @ T #Proyection matrix

    # Compute the 3D points
    X_3D = np.hstack((Op[5:].reshape(-1, 3), np.ones((nPoints, 1))))

    projection1 = theta_ext_1 @ X_3D.T
    projection1 = projection1[:2, :] / projection1[2, :]
    res1 = x1Data[:, :nPoints].T - projection1.T

    projection2 = theta_ext_2 @ X_3D.T
    projection2 = projection2[:2, :] / projection2[2, :]
    res2 = x2Data[:, :nPoints].T - projection2.T

    res = np.hstack((res1, res2)).flatten()

    return np.array(res)

def resBundleProjection_n_cameras(Op, xData, nCameras, K_c, nPoints):
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
    Op[0:1] -> theta, phi
    Op[2:5] -> Rx,Ry,Rz
    Op[6:7] -> theta, phi
    Op[8:11] -> Rx,Ry,Rz
    ...
    Op[] -> 3DXx,3DXy,3DXz
    '''
    # Bundle adjustment using least squares function
    idem = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
    # R = sc.linalg.expm(crossMatrix(Op[2:5]))
    # t = np.array([np.sin(Op[0])*np.cos(Op[1]), np.sin(Op[0])*np.sin(Op[1]), np.cos(Op[0])]).reshape(-1,1)
    theta_ext_1 = K_c @ idem
    # T = np.hstack((R, t))
    # theta_ext_2 =  K_c @ T #Proyection matrix

    theta_ext = []
    theta_ext.append(theta_ext_1)
    # theta_ext.append(theta_ext_2)
    for i in range(nCameras - 1):
        R = sc.linalg.expm(crossMatrix(Op[2+5*i:5+5*i]))
        t = np.array([np.sin(Op[5*i])*np.cos(Op[1+5*i]), np.sin(Op[0+5*i])*np.sin(Op[1+5*i]), np.cos(Op[0+5*i])]).reshape(-1,1)
        T = np.hstack((R, t))
        theta_ext.append(K_c @ T)

    # Compute the residuals
   
    Xpoints = xData
    # Xpoints = xData.reshape(nCameras, nPoints, 2)

    X_3D = np.hstack((Op[(nCameras-1)*5:].reshape(-1, 3), np.ones((nPoints, 1))))
    # print(X_3D)
    res = []
    for i in range(nCameras):
        projection = theta_ext[i] @ X_3D.T
        projection = projection[:2, :] / projection[2, :]
        res.append((Xpoints[i] - projection.T).flatten())

    # print(res)
    return np.array(res).flatten()

if __name__ == '__main__':

    image_pers_1 = cv2.imread('image1.png')
    image_pers_2 = cv2.imread('image2.png')
    image_pers_3 = cv2.imread('image3.png')

    # PART 2
    print("PART 2")
    # Load ground truth
    T_w_c1 = np.loadtxt('T_w_c1.txt')
    T_w_c2 = np.loadtxt('T_w_c2.txt')

    T_c1_w = np.linalg.inv(T_w_c1)
    T_c2_w = np.linalg.inv(T_w_c2)

    K_c = np.loadtxt('K_c.txt')

    F = np.loadtxt('F_21.txt')

    zeros = np.zeros(3).reshape(3,1)
    idem = np.hstack((np.identity(3),zeros))
    aux_matrix = np.dot(K_c,idem)
    P1 = np.dot(aux_matrix,T_c1_w) 
    P2 = np.dot(aux_matrix,T_c2_w)

    # Load points
    points1 = np.loadtxt('x1Data.txt')
    points2 = np.loadtxt('x2Data.txt')
    points3 = np.loadtxt('x3Data.txt')
    worldPoints = np.loadtxt('X_w.txt')

    E_24 = compute_essential_matrix(F, K_c)
    solutions_24 = decompose_essential_matrix(points1, points2, E_24, K_c)
    # add last row to make it 4x4
    solutions_24 = np.vstack((solutions_24, np.array([0, 0, 0, 1])))
    

    T_w_c1_24 = np.linalg.inv(K_c) @ P1
    T_w_c2_24 = np.linalg.inv(K_c) @ P2

    # Compute the 3D points
    # points3D_24 = triangulation(T_w_c1_24, T_w_c2_24, points1, points2)
    # points3D_24 = points3D_24.T

    
    ##Plot the 3D cameras and the 3D points
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')
    drawRefSystem(ax, T_w_c1 @ np.linalg.inv(solutions_24), '-', 'C2 estimated')

    ax.scatter(worldPoints[0, :], worldPoints[1, :], worldPoints[2, :], marker='.')
    # ax.scatter(points3D_24[0, :], points3D_24[1, :], points3D_24[2, :], marker='.', color='r')
    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    ax.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    
    plt.show()

    # PART 2.1
   
    print("PART 2.1")
    Op = [0.0, 0.0, 0.0, 0.0, 0.0] + worldPoints[0:3].T.flatten().tolist()
    
    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(points1, points2, K_c, points1.shape[1]), method='lm')
    R_c2_c1 = sc.linalg.expm(crossMatrix(OpOptim.x[2:5]))
    t_c2_c1 = np.array([np.sin(OpOptim.x[0])*np.cos(OpOptim.x[1]), np.sin(OpOptim.x[0])*np.sin(OpOptim.x[1]), np.cos(OpOptim.x[0])]).reshape(-1,1)
    T_c2_c1_op = np.hstack((R_c2_c1, t_c2_c1))
    P2_op = K_c @ T_c2_c1_op
    T_c2_c1_op = np.vstack((T_c2_c1_op, np.array([0, 0, 0, 1])))
    points_3D_Op = np.concatenate((OpOptim.x[5: 8], np.array([1.0])), axis=0)

    for i in range(worldPoints.shape[1]-1):
        points_3D_Op = np.vstack((points_3D_Op, np.concatenate((OpOptim.x[8+3*i: 8+3*i+3], np.array([1.0])) ,axis=0)))

    #### Draw 3D ################
    fig3D = plt.figure(2)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    #drawRefSystem(ax, wTc1 @ np.linalg.inv(c2Tc1_Op), '-', 'C2_BA')
    drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c2_c1_op), '-', 'C2_BA_scaled')
    drawRefSystem(ax, T_w_c2, '-', 'C2_True')

    points_Op = T_w_c1 @ (points_3D_Op).T

    ax.scatter(points_Op[0, :], points_Op[1, :], points_Op[2, :], marker='.')
    # plotNumbered3DPoints(ax, points_Op, 'b', 0.1)

    ax.scatter(worldPoints[0, :], worldPoints[1, :], worldPoints[2, :], marker='.')
    # plotNumbered3DPoints(ax, worldPoints, 'r', 0.1)

    plt.title('3D points Bundle adjustment (red=True data)')
    plt.show()

    #### Plot residual bundel adj ##############
    idem = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
    P1_est = K_c @ idem
    x1_p = P1_est @ points_3D_Op.T
    x1_p = x1_p / x1_p[2, :]
    x2_p = P2_op @ points_3D_Op.T
    x2_p = x2_p / x2_p[2, :]

    plt.figure(4)
    plt.imshow(image_pers_1, cmap='gray', vmin=0, vmax=255)
    plt.title('Residuals after Bundle adjustment Image1')
    
    plotResidual(points1.T, x1_p.T, 'k-')
 
    plt.draw()

    plt.show()

    plt.figure(5)
    plt.imshow(image_pers_2, cmap='gray', vmin=0, vmax=255)
    plt.title('Residuals after Bundle adjustment Image2')
    plotResidual(points2.T, x2_p.T, 'k-')

    plt.draw()

    plt.show()

    # PART 3

    print("PART 3")

    T_w_c3 = np.loadtxt('T_w_c3.txt')

    # PNP pose estimation of camera 3
    points_3D_Scene = np.float32(points_3D_Op[:, 0:3])
    points_C3 = np.ascontiguousarray(points3[0:2, :].T).reshape((points3.shape[1], 1, 2))
    Coeff=[]
    retval, rvec, tvec = cv2.solvePnP(objectPoints=points_3D_Scene, imagePoints=points_C3, cameraMatrix=K_c,
                                      distCoeffs=np.array(Coeff), flags=cv2.SOLVEPNP_EPNP)
    
    R_c3_c1_pnp = sc.linalg.expm(crossMatrix(rvec))
    t_c3_c1_pnp = tvec
    T_c3_c1_pnp = np.hstack((R_c3_c1_pnp, t_c3_c1_pnp))
    T_c3_c1_pnp = np.vstack((T_c3_c1_pnp, np.array([0, 0, 0, 1])))

    fig3D = plt.figure(8)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1_True')
    drawRefSystem(ax, T_w_c3, '-', 'C3_True')
    drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c3_c1_pnp), '-', 'C3_PnP')
    plt.title('3D camera poses PnP')
    plt.draw()
    plt.show()

    # PART 4
    # print("PART 4")

    # Op = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + worldPoints[0:3].T.flatten().tolist()

    # X = np.vstack((np.vstack((points1, points2)), points3))
    # OpOptim = scOptim.least_squares(resBundleProjection_n_cameras, Op, args=(X, 3, K_c, points1.shape[1]), method='lm')

    Op = [0.0, 0.0, 0.0, 0.0, 0.0] + worldPoints[0:3].T.flatten().tolist()

    X = np.stack((points1.T, points2.T))
    OpOptim = scOptim.least_squares(resBundleProjection_n_cameras, Op, args=(X, 2, K_c, points1.shape[1]), method='lm')

    print(T_c2_c1_op)
    R_c2_c1 = sc.linalg.expm(crossMatrix(OpOptim.x[2:5]))
    t_c2_c1 = np.array([np.sin(OpOptim.x[0])*np.cos(OpOptim.x[1]), np.sin(OpOptim.x[0])*np.sin(OpOptim.x[1]), np.cos(OpOptim.x[0])]).reshape(-1,1)
    T_c2_c1_op = np.hstack((R_c2_c1, t_c2_c1))
    P2_op = K_c @ T_c2_c1_op
    T_c2_c1_op = np.vstack((T_c2_c1_op, np.array([0, 0, 0, 1])))
    print(T_c2_c1_op)

    # R_c3_c1 = sc.linalg.expm(crossMatrix(OpOptim.x[7:10]))
    # t_c3_c1 = np.array([np.sin(OpOptim.x[5])*np.cos(OpOptim.x[6]), np.sin(OpOptim.x[5])*np.sin(OpOptim.x[6]), np.cos(OpOptim.x[5])]).reshape(-1,1)
    # T_c3_c1_op = np.hstack((R_c3_c1, t_c3_c1))
    # P3_op = K_c @ T_c3_c1_op
    # T_c3_c1_op = np.vstack((T_c3_c1_op, np.array([0, 0, 0, 1])))

    # points_3D_Op = np.concatenate((OpOptim.x[10: 10+3], np.array([1.0])), axis=0)

    # for i in range(worldPoints.shape[1]-1):
    #     points_3D_Op = np.vstack((points_3D_Op, np.concatenate((OpOptim.x[13+3*i: 13+3*i+3], np.array([1.0])) ,axis=0)))

    for i in range(worldPoints.shape[1]-1):
        points_3D_Op = np.vstack((points_3D_Op, np.concatenate((OpOptim.x[8+3*i: 8+3*i+3], np.array([1.0])) ,axis=0)))


    #### Draw 3D ################
    fig3D = plt.figure(2)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    #drawRefSystem(ax, wTc1 @ np.linalg.inv(c2Tc1_Op), '-', 'C2_BA')
    drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c2_c1_op), '-', 'C2_BA_scaled')
    drawRefSystem(ax, T_w_c2, '-', 'C2_True')
    # drawRefSystem(ax, T_w_c1 @ np.linalg.inv(T_c3_c1_op), '-', 'C3_BA_scaled')
    drawRefSystem(ax, T_w_c3, '-', 'C3_True')

    points_Op = T_w_c1 @ (points_3D_Op).T

    # ax.scatter(points_Op[0, :], points_Op[1, :], points_Op[2, :], marker='.')
    # plotNumbered3DPoints(ax, points_Op, 'b', 0.1)

    ax.scatter(worldPoints[0, :], worldPoints[1, :], worldPoints[2, :], marker='.')
    # plotNumbered3DPoints(ax, worldPoints, 'r', 0.1)

    plt.title('3D points Bundle adjustment (red=True data)')
    plt.show()


