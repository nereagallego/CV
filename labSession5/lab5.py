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


