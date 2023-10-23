import matplotlib.pyplot as plt
import numpy as np
import cv2
import math



# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c


def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)


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
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)


def plotLabelled3DPoints(ax, X, labels, strColor, offset):
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
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

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

def drawLine(l,strFormat,lWidth, label=""):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.array([0, -l[2] / l[1]])
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.array([-l[2] / l[0], 0])
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)
   

def line_intersection(line1, line2):
    y = (line1[0] * line2[2] - line1[2]*line2[0]) / (line1[1]*line2[0]-line1[0]*line2[1])
    x = (line1[2]*line2[1] - line1[1]*line2[2]) / (line1[1] * line2[0] - line1[0] * line2[1])
    return np.array([x, y])

def distance(plane, point):
    return abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]) / math.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2])

def plotPoint(point, label):
    plt.plot(point[0], point[1], '+r')
    plt.annotate(label,(point[0], point[1]), color='r',textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center')
    


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


# Compute the fundamental matrix from ground truth poses
    """_summary_ neew to check this
    """
def compute_fundamental_matrix_from_poses(T_c1_w, T_c2_w):
    # Compute the essential matrix
    T_c2_c1 = np.dot(T_c2_w, np.linalg.inv(T_c1_w))
    R_c2_c1 = T_c2_c1[0:3, 0:3]
    t_c2_c1 = T_c2_c1[0:3, 3]
    t_c2_c1_x = np.array([[0, -t_c2_c1[2], t_c2_c1[1]], [t_c2_c1[2], 0, -t_c2_c1[0]], [-t_c2_c1[1], t_c2_c1[0], 0]])
    E = t_c2_c1_x @ R_c2_c1
    # Compute the fundamental matrix
    K_c = np.loadtxt('K_c.txt')
    F = np.dot(np.linalg.inv(K_c).T, E) @ np.linalg.inv(K_c)
    return F/F[2,2]

# Compute the epipole from the fundamental matrix
def compute_epipole(F):
    # Compute the epipole
    _, _, V = np.linalg.svd(F)
    e = V[-1, :]
    e = e / e[-1]
    return e


def compute_fundamental_matrix(points1, points2):
    """_summary_ neew to check this"""
    # Compute the fundamental matrix
    A = np.zeros((points1.shape[1], 9))
    for i in range(points1.shape[1]):
        A[i, :] = [points1[0, i] * points2[0, i], points2[0, i] * points1[1, i], points2[0, i], points1[0, i] * points2[1, i], points1[1, i] * points2[1, i], points2[1,i], points1[0,i], points1[1,i], 1]

    # compute linear least squares solution
    _, _, V = np.linalg.svd(A)
    F = V[-1, :].reshape(3, 3)

    return F/F[2,2]

def show_epipolar_lines(img1, img2, T_c1_w, T_c2_w, F):
    
    # Compute the epipole
    epipole = compute_epipole(F)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.imshow(img1)
    ax2.imshow(img2)

    plt.show(block=False) 

    print('Click on the figure to select points...')
    i = 0
    while i < 8:
        # Wait for the user to click on the figure
        clicked_point = fig1.ginput(n=1, timeout=0)
        # print(clicked_point)
        ax1.scatter(clicked_point[0][0], clicked_point[0][1], c='r', s=40)
        fig1.canvas.draw()

        # Compute the epipolar line
        x1 = np.array([clicked_point[0][0], clicked_point[0][1]])
        line = compute_epipolar_line(x1, F) 
        y = int(-line[2]/line[1])
        x = int(-line[2]/line[0])
        ax2.plot([x, 0], [0, y], c='b', linewidth=1)
        fig2.canvas.draw()

        i += 1

    #Draw the epipole
    ax2.scatter(epipole[0], epipole[1], c='g', s=40)
    fig2.canvas.draw()
    
    # Add key press event handler to close figures on ESC key press
    def on_key_press(event):
        if event.key == 'escape':
            plt.close(fig1)
            plt.close(fig2)

    fig1.canvas.mpl_connect('key_press_event', on_key_press)
    fig2.canvas.mpl_connect('key_press_event', on_key_press)
    
    print('Press ESC to close the figures...')
    plt.show(block=True)


def compute_epipolar_line(x1, F):
    # Convert clicked point to homogeneous coordinates
    x1 = np.append(x1, 1)

    # Compute the epipolar line
    l = np.dot(F, x1)

    # Normalize the line
    l = l / np.linalg.norm(l)

    return l


clicked_coordinates = []

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coordinates.append((x, y))
        # print(f"Clicked at pixel coordinates: ({x}, {y})")


def plotPoint(point, label):
    plt.plot(point[0], point[1], '+r')
    plt.annotate(label,(point[0], point[1]), color='r',textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center')
    
def drawLine(l,strFormat,lWidth, label=""):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.array([0, -l[2] / l[1]])
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.array([-l[2] / l[0], 0])
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)

# Compute the Essential matrix from Fundamental matrix
def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    return E

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

def computeHomography(points1, points2):
    A = np.zeros((points1.shape[1] * 2, 9))
    for i in range(points1.shape[1]):
        A[2*i, :] = [points1[0,i], points1[1,i], 1, 0, 0, 0, -points2[0,i]*points1[0,i], -points2[0,i]*points1[1,i], -points2[0,i]]
        A[2*i+1,:] = [0, 0, 0, points1[0,i], points1[1,i], 1, -points2[1,i]*points1[0,i], -points2[1,i]*points1[1,i], -points2[1,i]]
    
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape(3, 3)
    return H/H[2,2]

def generate_random_points_on_plane(plane_coefficients, num_points, x_range=(-10, 10), y_range=(-10, 10)):
    a, b, c, d = plane_coefficients
    x_values = np.random.uniform(low=x_range[0], high=x_range[1], size=num_points)
    y_values = np.random.uniform(low=y_range[0], high=y_range[1], size=num_points)
    z_values = -(a * x_values + b * y_values + d) / c
    random_points = np.column_stack((x_values, y_values, z_values))
    return random_points
   
if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    T_w_c1 = np.loadtxt('T_w_c1.txt')
    T_w_c2 = np.loadtxt('T_w_c2.txt')

    T_c1_w = np.linalg.inv(T_w_c1)
    T_c2_w = np.linalg.inv(T_w_c2)

    K_c = np.loadtxt('K_c.txt')

    zeros = np.zeros(3).reshape(3,1)
    idem = np.hstack((np.identity(3),zeros))
    aux_matrix = np.dot(K_c,idem)
    P1 = np.dot(aux_matrix,T_c1_w) 
    P2 = np.dot(aux_matrix,T_c2_w)

    # Load points
    points1 = np.loadtxt('x1Data.txt')
    points2 = np.loadtxt('x2Data.txt')
    worldPoints = np.loadtxt('X_w.txt')

    #PART 1
    
    triangulation(P1, P2, points1, points2)

    #Now we wnat to see the matching point in each image
    img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)

    plt.figure(figsize =(17,7))
    plt.subplot(1,2,1)
    plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
    plt.plot(points1[0, :], points1[1, :],'rx', markersize=10)
    plotNumberedImagePoints(points1, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.plot(points2[0, :], points2[1, :],'rx', markersize=10)
    plotNumberedImagePoints(points2, 'r', (10,0)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 2')
    plt.draw()

    #PART 2

    print("PART 2.1")
    F_21 = np.loadtxt('F_21_test.txt')

    img1 = cv2.cvtColor(cv2.imread("image1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("image2.png"), cv2.COLOR_BGR2RGB)

    show_epipolar_lines(img1, img2, T_c1_w, T_c2_w, F_21)
    
    # PART 2.2
    print("PART 2.2")
    F_22 = compute_fundamental_matrix_from_poses(T_c1_w, T_c2_w)

    show_epipolar_lines(img1, img2, T_c1_w, T_c2_w, F_22)


    # PART 2.3
    print("PART 2.3")
    F_23 = compute_fundamental_matrix(points1, points2)

    show_epipolar_lines(img1, img2, T_c1_w, T_c2_w, F_23)

    # PART 2.4
    print("PART 2.4")

    E_24 = compute_essential_matrix(F_21, K_c)
    solutions_24 = decompose_essential_matrix(points1, points2, E_24, K_c)
    # add last row to make it 4x4
    solutions_24 = np.vstack((solutions_24, np.array([0, 0, 0, 1])))
    print(solutions_24)
 
    # PART 2.5
    
    # Visualize the cameras and the 3D points
    T_w_c1_24 = np.linalg.inv(K_c) @ P1
    T_w_c2_24 = np.linalg.inv(K_c) @ P2

    # print("own")
    # print(T_w_c1_24)
    # print(T_w_c2_24)

    # print("gt")
    # print(T_w_c1)
    # print(T_w_c2)
    
    ##Plot the 3D cameras and the 3D points
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # =============
    # First subplot
    # =============
    # set up the axes for the first plot
    ax = fig.add_subplot(1, 1, 1, projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')
    # drawRefSystem(ax, T_w_c1_24, '-', 'C1 estimated')
    drawRefSystem(ax, T_w_c1 @ np.linalg.inv(solutions_24), '-', 'C2 estimated')

    ax.scatter(worldPoints[0, :], worldPoints[1, :], worldPoints[2, :], marker='.')
    # plotNumbered3DPoints(ax, worldPoints, 'r', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    ax.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    
    plt.show()


    # PART 3.1
    print("PART 3.1")
    # From the provided poses and the plane equation coefficients on camera reference 1, compute the homography that relates both images through the floor plane
    Pi_1 = np.array([[0.0149, 0.9483, 0.3171, -1.7257]])

    T_c2_c1 = np.dot(T_c2_w, np.linalg.inv(T_c1_w))
    R_c2_c1 = T_c2_c1[0:3, 0:3]
    t_c2_c1 = T_c2_c1[0:3, 3:4]

    # Compute the homography
    H = K_c @ (R_c2_c1 - t_c2_c1 @ Pi_1[:,0:3] / Pi_1[0,3])@ np.linalg.inv(K_c)
    H = H/H[2,2]
   

    point = H @ np.array([414,375,1]).T

    print(point/point[2])


    print(H)


    # PART 3.2
    print("PART 3.2")

    matches1 = np.loadtxt('x1FloorData.txt')
    matches2 = np.loadtxt('x2FloorData.txt')
    matches1 = matches1/matches1[2,:]

    matches1_new = H@matches1
    matches1_new /= matches1_new[2,:]

    plt.figure(figsize =(17,7))
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.plot(matches1[0, :], matches1[1, :],'rx', markersize=10)
    plt.title('Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.plot(matches1_new[0, :], matches1_new[1,:],'rx', markersize=10)
    plt.title('Image 2')
    plt.show()
    
    # PART 3.3
    print("PART 3.3")

    # Compute the homography
    H = computeHomography(matches1, matches2)

    point = H @ np.array([414,375,1]).T
    print(point/point[2])

    print(H)

    # Propose and use a metric for evaluating the accuracy of your results.
    # Compute the error
    error = 0
    for i in range(matches1.shape[1]):
        # mean square error
        error += math.sqrt(np.linalg.norm(matches2[:,i] - H @ matches1[:,i])**2)
    print(error)
