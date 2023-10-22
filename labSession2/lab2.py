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
    
"""
    Triangulate a set of points given the projection matrices of two cameras.
"""
def triangulation(P1, P2, points1, points2, worldPoints):
    point = 0
    while point < points1.shape[1]:
        p1 = points1[:, point]
        p2 = points2[:, point]
        pw = worldPoints[:, point]
        # Triangulate the points
        A = [p1[0] * P1[2, :] - P1[0, :], p1[1] * P1[2, :] - P1[1, :], p2[0] * P2[2, :] - P2[0, :], p2[1] * P2[2, :] - P2[1, :]]
        _, _, V = np.linalg.svd(A)
        X = V[-1, :]
        X = X / X[-1]
        if abs(X[0] - pw[0]) > 0.2 or abs(X[1] - pw[1]) > 0.2 or abs(X[2] - pw[2]) > 0.2:
            print("Error in triangulation")
        point += 1

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
    # print(points1.shape)
    # print(points2.shape)
    # Compute the fundamental matrix
    A = np.zeros((points1.shape[1], 9))
    for i in range(points1.shape[1]):
        A[i, :] = [points1[0, i] * points2[0, i], points2[0, i] * points1[1, i], points2[0, i], points1[0, i] * points2[1, i], points1[1, i] * points2[1, i], points2[1,i], points1[0,i], points1[1,i], 1]

    # compute linear least squares solution
    _, _, V = np.linalg.svd(A)
    F = V[-1, :].reshape(3, 3)

    return F/F[2,2]

def show_part22(img1, img2, T_c1_w, T_c2_w):
    F_22 = compute_fundamental_matrix_from_poses(T_c1_w, T_c2_w)
    epipole_22 = compute_epipole(F_22)

    clicked_coordinates = []

    cv2.imshow


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
    return E/E[2,2]

# Decompose the Essential matrix
def decompose_essential_matrix(E):
    # Compute the SVD of the essential matrix
    U, _, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Compute the four possible solutions
    solutions = []
    solutions.append((U @ W @ V, U[:, 2], V[:, 2]))
    solutions.append((U @ W @ V, -U[:, 2], V[:, 2]))
    solutions.append((U @ W.T @ V, U[:, 2], -V[:, 2]))
    solutions.append((U @ W.T @ V, -U[:, 2], -V[:, 2]))
    
    

    
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
    
    triangulation(P1, P2, points1, points2, worldPoints)

    #PART 2

    F_21 = np.loadtxt('F_21_test.txt')
    print(F_21)
    
    # Global variables to store coordinates
    current = -1

    img1 = cv2.cvtColor(cv2.imread("image1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("image2.png"), cv2.COLOR_BGR2RGB)


    colors = ['blue', 'red', 'green', 'black', 'orange', 'grey', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow',  'violet', 'indigo', 'lightblue']

    while True:
        # conc = np.concatenate((img1, img2), axis=1)
        # Display the resulting frame
        cv2.imshow('Image 1', img1)
        plt.figure(1)
        plt.imshow(img1)

        for i in range(len(clicked_coordinates)):
            cv2.circle(img1, clicked_coordinates[i], 5, (0, 0, 255), -1)
            # plotPoint(clicked_coordinates[i], str(i))

        cv2.setMouseCallback('Image 1', mouse_callback)

        plt.draw()

        cv2.imshow('Image 2', img2)

        for i in range(len(clicked_coordinates)):
            line = compute_epipolar_line(clicked_coordinates[i], F_21)
            # drawLine(line, colors[i], 1)
            y = int(-line[2]/line[0])
            x = int(-line[2]/line[1])
            cv2.line(img2, (0,y), (x,0),(255,0,0), 2)
        
        if cv2.waitKey(10) & 0xFF == 27:  # Break the loop on ESC key
            # close all open windows
            cv2.destroyAllWindows()
            break
    
    # PART 2.2
    print("PART 2.2")
    F_22 = compute_fundamental_matrix_from_poses(T_c1_w, T_c2_w)
    print(F_22)
    epipole_22 = compute_epipole(F_22)
    print(epipole_22)

    
    img1 = cv2.cvtColor(cv2.imread("image1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("image2.png"), cv2.COLOR_BGR2RGB)

    

    clicked_coordinates = []

    while True:
        # conc = np.concatenate((img1, img2), axis=1)
        # Display the resulting frame
        cv2.imshow('Image 1', img1)
        plt.figure(1)
        plt.imshow(img1)

        for i in range(len(clicked_coordinates)):
            cv2.circle(img1, clicked_coordinates[i], 5, (0, 0, 255), -1)
            # plotPoint(clicked_coordinates[i], str(i))

        cv2.setMouseCallback('Image 1', mouse_callback)

        plt.draw()

        cv2.imshow('Image 2', img2)

        for i in range(len(clicked_coordinates)):
            line = compute_epipolar_line(clicked_coordinates[i], F_22)
            # drawLine(line, colors[i], 1)
            y = int(-line[2]/line[0])
            x = int(-line[2]/line[1])
            cv2.line(img2, (0,y), (x,0),(255,0,0), 2)

        # Create a larger canvas to accommodate the point
        canvas_width, canvas_height = max(img2.shape[0], epipole_22[0] + 1), max(img2.shape[1], epipole_22[1] + 1)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Copy the original image onto the canvas
        canvas[0:img2.shape[0], 0:img2.shape[1]] = img2
        
        # Draw the epipole
        cv2.circle(canvas, (int(epipole_22[0]), int(epipole_22[1])), 5, (0, 0, 255), -1)

        if cv2.waitKey(10) & 0xFF == 27:  # Break the loop on ESC key
            # close all open windows
            cv2.destroyAllWindows()
            break
    

    # PART 2.3
    print("PART 2.3")
    F_23 = compute_fundamental_matrix(points1, points2)
    print(F_23)

    img1 = cv2.cvtColor(cv2.imread("image1.png"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("image2.png"), cv2.COLOR_BGR2RGB)

    clicked_coordinates = []

    while True:
        # conc = np.concatenate((img1, img2), axis=1)
        # Display the resulting frame
        cv2.imshow('Image 1', img1)
        plt.figure(1)
        plt.imshow(img1)

        for i in range(len(clicked_coordinates)):
            cv2.circle(img1, clicked_coordinates[i], 5, (0, 0, 255), -1)
            # plotPoint(clicked_coordinates[i], str(i))

        cv2.setMouseCallback('Image 1', mouse_callback)

        plt.draw()

        cv2.imshow('Image 2', img2)

        for i in range(len(clicked_coordinates)):
            line = compute_epipolar_line(clicked_coordinates[i], F_23)
            # drawLine(line, colors[i], 1)
            y = int(-line[2]/line[0])
            x = int(-line[2]/line[1])
            cv2.line(img2, (0,y), (x,0),(255,0,0), 2)
        
        if cv2.waitKey(10) & 0xFF == 27:  # Break the loop on ESC key
            # close all open windows
            cv2.destroyAllWindows()
            break

    # PART 2.4
    print("PART 2.4")

    E_21 = compute_essential_matrix(F_21, K_c)
    
    
    