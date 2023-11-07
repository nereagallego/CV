""""
    Implemetation for lab 4 session
    author: César Borja Moreno and Nerea Gallego Sánchez
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.optimize as scOptim

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
    Op[0:2] -> theta, phi
    Op[2:5] -> Rx,Ry,Rz
    Op[5:5 + nPoints*3] -> 3DXx,3DXy,3DXz
    '''
    # Bundle adjustment using least squares function
    idem = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
    R = sc.linalg.expm(crossMatrix(Op[2:5]))
    t = np.array([np.sin(Op[0])*np.cos(Op[1]), np.sin(Op[0])*np.sin(Op[1]), np.cos(Op[0])]).reshape(-1,1)
    theta_ext_1 = K_c @ idem
    T = np.hstack((R, t))
    # T = np.vstack((T, np.array([0, 0, 0, 1])))
    theta_ext_2 =  K_c @ T #Proyection matrix

    # Compute the 3D points
    res = []
    for i in range(nPoints):

        X_3D = np.hstack((Op[5+i*3: 5+i*3+3], np.array([1.0])))
        projection1 = theta_ext_1 @ X_3D
        projection1 = projection1[0:2] / projection1[2]
        x1 = x1Data[0:1, i]
        res1 = x1 - projection1

        projection2 = theta_ext_2 @ X_3D
        projection2 = projection2[0:2] / projection2[2]
        x2 = x2Data[0:1, i]
        res2 = x2 - projection2

        res.append(res1[0])
        res.append(res1[1])
        res.append(res2[0])
        res.append(res2[1])

    return np.array(res)

if __name__ == '__main__':
    print("Hello world")

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
    worldPoints = np.loadtxt('X_w.txt')

    E_24 = compute_essential_matrix(F, K_c)
    solutions_24 = decompose_essential_matrix(points1, points2, E_24, K_c)
    # add last row to make it 4x4
    solutions_24 = np.vstack((solutions_24, np.array([0, 0, 0, 1])))
    print(solutions_24)

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
    Op = [0.0, 0.0, 0.0, 0.0, 0.0] + worldPoints[0:3].flatten().tolist()
    OpOptim = scOptim.least_squares(resBundleProjection, Op, args=(points1, points2, K_c, points1.shape[1]), method='lm')




