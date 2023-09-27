from mpl_toolkits.mplot3d import Axes3D

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
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth, label=label)

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
    plotPoint(X_A_c1,'A')
    plotPoint(X_B_c1, 'B')
    plotPoint(X_C_c1,'C')
    plotPoint(X_D_c1, 'D')
    plotPoint(X_E_c1, 'E')
   
    drawLine(line_ab_c1,'g-',1)
    drawLine(line_cd_c1,'g-',1)
    plt.plot(p_12_c1[0], p_12_c1[1], marker="o", markersize=2,  markerfacecolor="blue")
    plt.plot(AB_inf_c1[0], AB_inf_c1[1], marker="o", markersize=2,  markerfacecolor="blue")
    plt.draw()
    plt.waitforbuttonpress()
    img2 = cv2.cvtColor(cv2.imread("Image2.jpg"), cv2.COLOR_BGR2RGB)
    plt.figure(2)
    plt.imshow(img2)
    plotPoint(X_A_c2,'A')
    plotPoint(X_B_c2, 'B')
    plotPoint(X_C_c2,'C')
    plotPoint(X_D_c2, 'D')
    plotPoint(X_E_c2, 'E')
    drawLine(line_ab_c2,'g-',1)
    drawLine(line_cd_c2,'g-',1)
    plt.plot(p_12_c2[0], p_12_c2[1], marker="o", markersize=2,  markerfacecolor="blue")
    plt.draw()
    plt.waitforbuttonpress()

    
    # PART 3

    X_w = np.array((X_A_h,X_B_h,X_C_h,X_D_h))
    A = (X_w[:,0:4])
    U, s, vh = np.linalg.svd(A)
    plane = np.reshape(vh[-1,:],(4,1))
    
    plane = plane.reshape(4)
    print(plane)

    print(distance(plane,X_A))
    print(distance(plane,X_B))
    print(distance(plane,X_C))
    print(distance(plane,X_D))
    print(distance(plane,X_E))
