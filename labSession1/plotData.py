#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 1
#
# Title: 2D-3D geometry in homogeneous coordinates and camera projection
#
# Date: 14 September 2022
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2



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

def drawLine(l,strFormat,lWidth):
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

if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    # Load ground truth
    R_w_c1 = np.loadtxt('R_w_c1.txt')
    R_w_c2 = np.loadtxt('R_w_c2.txt')

    t_w_c1 = np.loadtxt('t_w_c1.txt')
    t_w_c2 = np.loadtxt('t_w_c2.txt')

    T_w_c1 = ensamble_T(R_w_c1, t_w_c1)
    T_w_c2 = ensamble_T(R_w_c2, t_w_c2)
    T_c1_w = np.linalg.inv(T_w_c1)
    T_c2_w = np.linalg.inv(T_w_c2)

    K_c = np.loadtxt('K.txt')

    zeros = np.zeros(3).reshape(3,1)
    idem = np.hstack((np.identity(3),zeros))
    aux_matrix = np.dot(K_c,idem)
    P1 = np.dot(aux_matrix,T_c1_w) 
    P2 = np.dot(aux_matrix,T_c2_w)
    print(P1)    




    X_A = np.array([3.44, 0.80, 0.82])
    X_B = np.array([4.20, 0.80, 0.82])
    X_C = np.array([4.20, 0.60, 0.82])
    X_D = np.array([3.55, 0.60, 0.82])
    X_E = np.array([-0.01, 2.6, 1.21])

    X_A_h = np.array([3.44, 0.80, 0.82, 1])
    X_B_h = np.array([4.20, 0.80, 0.82, 1])
    X_C_h = np.array([4.20, 0.60, 0.82, 1])
    X_D_h = np.array([3.55, 0.60, 0.82, 1])
    X_E_h = np.array([-0.01, 2.6, 1.21, 1])

    X_A_c1 = np.dot(P1, X_A_h)
    X_B_c1 = np.dot(P1, X_B_h)
    X_C_c1 = np.dot(P1, X_C_h)
    X_D_c1 = np.dot(P1, X_D_h)
    X_E_c1 = np.dot(P1, X_E_h)

    X_A_c1 = [X_A_c1[0]/X_A_c1[2], X_A_c1[1]/X_A_c1[2], X_A_c1[2]/X_A_c1[2]]
    X_B_c1 = [X_B_c1[0]/X_B_c1[2], X_B_c1[1]/X_B_c1[2], X_B_c1[2]/X_B_c1[2]]
    X_C_c1 = [X_C_c1[0]/X_C_c1[2], X_C_c1[1]/X_C_c1[2], X_C_c1[2]/X_C_c1[2]]
    X_D_c1 = [X_D_c1[0]/X_D_c1[2], X_D_c1[1]/X_D_c1[2], X_D_c1[2]/X_D_c1[2]]
    X_E_c1 = [X_E_c1[0]/X_E_c1[2], X_E_c1[1]/X_E_c1[2], X_E_c1[2]/X_E_c1[2]]

    X_A_c2 = np.dot(P2, X_A_h)
    X_B_c2 = np.dot(P2, X_B_h)
    X_C_c2 = np.dot(P2, X_C_h)
    X_D_c2 = np.dot(P2, X_D_h)
    X_E_c2 = np.dot(P2, X_E_h)

    X_A_c2 = [X_A_c2[0]/X_A_c2[2], X_A_c2[1]/X_A_c2[2], X_A_c2[2]/X_A_c2[2]]
    X_B_c2 = [X_B_c2[0]/X_B_c2[2], X_B_c2[1]/X_B_c2[2], X_B_c2[2]/X_B_c2[2]]
    X_C_c2 = [X_C_c2[0]/X_C_c2[2], X_C_c2[1]/X_C_c2[2], X_C_c2[2]/X_C_c2[2]]
    X_D_c2 = [X_D_c2[0]/X_D_c2[2], X_D_c2[1]/X_D_c2[2], X_D_c2[2]/X_D_c2[2]]
    X_E_c2 = [X_E_c2[0]/X_E_c2[2], X_E_c2[1]/X_E_c2[2], X_E_c2[2]/X_E_c2[2]]

    print(X_A_c2)
    print(X_B_c2)
    print(X_C_c2)
    print(X_D_c2)
    print(X_E_c2)


    #PART 2

    line_ab_c1 = np.cross(X_A_c1,X_B_c1)
    line_ab_c2 = np.cross(X_A_c2,X_B_c2)
    print(line_ab_c1)
    print(line_ab_c2)

    line_cd_c1 = np.cross(X_C_c1,X_D_c1)
    line_cd_c2 = np.cross(X_C_c2,X_D_c2)

    p_12_c1 = line_intersection(line_ab_c1,line_cd_c1)
    p_12_c2 = line_intersection(line_ab_c2,line_cd_c2)

    AB_inf = np.array([X_A[0] - X_B[0], X_A[1]-X_B[1], X_A[2]-X_B[2], 0])
    AB_inf_c1 = np.dot(P1,AB_inf)
    AB_inf_c1 = np.array([AB_inf_c1[0]/AB_inf_c1[2], AB_inf_c1[1]/AB_inf_c1[2], AB_inf_c1[2]/AB_inf_c1[2]])

    img1 = cv2.cvtColor(cv2.imread("Image1.jpg"), cv2.COLOR_BGR2RGB)
    plt.figure(1)
    plt.imshow(img1)
    drawLine(line_ab_c1,'g-',1)
    drawLine(line_cd_c1,'g-',1)
    plt.plot(p_12_c1[0], p_12_c1[1], marker="o", markersize=2,  markerfacecolor="blue")
    plt.plot(AB_inf_c1[0], AB_inf_c1[1], marker="o", markersize=2,  markerfacecolor="blue")
    plt.draw()
    img2 = cv2.cvtColor(cv2.imread("Image2.jpg"), cv2.COLOR_BGR2RGB)
    plt.figure(2)
    plt.imshow(img2)
    drawLine(line_ab_c2,'g-',1)
    drawLine(line_cd_c2,'g-',1)
    plt.plot(p_12_c2[0], p_12_c2[1], marker="o", markersize=2,  markerfacecolor="blue")
    plt.draw()

    




    print(np.array([[3.44, 0.80, 0.82]]).T) #transpose need to have dimension 2
    print(np.array([3.44, 0.80, 0.82]).T) #transpose does not work with 1 dim arrays

    # Example of transpose (need to have dimension 2)  and concatenation in numpy
    X_w = np.vstack((np.hstack((np.reshape(X_A,(3,1)), np.reshape(X_C,(3,1)))), np.ones((1, 2))))

    ##Plot the 3D cameras and the 3D points
    fig3D = plt.figure(3)

    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    plotNumbered3DPoints(ax, X_w, 'r', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)
    plotLabelled3DPoints(ax, X_w, ['A','C'], 'r', (-0.3, -0.3, 0.1)) # For plotting with labels (choose one of the both options)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    #Drawing a 3D segment
    draw3DLine(ax, X_A, X_C, '--', 'k', 1)

    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()

    ## 2D plotting example
    img1 = cv2.cvtColor(cv2.imread("Image1.jpg"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("Image2.jpg"), cv2.COLOR_BGR2RGB)

    x1 = np.array([[527.7253,334.1983],[292.9017,392.1474]])

    plt.figure(1)
    plt.imshow(img1)
    plt.plot(x1[0, :], x1[1, :],'+r', markersize=15)
    plotLabeledImagePoints(x1, ['a','c'], 'r', (20,-20)) # For plotting with labels (choose one of the both options)
    plotNumberedImagePoints(x1, 'r', (20,25)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
