#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: Optical Flow
#
# Date: 22 November 2020
#
#####################################################################################
#
# Authors: Jose Lamarca, Jesus Bermudez, Richard Elvira, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def read_image(filename: str, ):
    """
    Read image using opencv converting from BGR to RGB
    :param filename: name of the image
    :return: np matrix with the image
    """
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def read_flo_file(filename, verbose=False):
    """
    Read from .flo optical flow file (Middlebury format)
    :param flow_file: name of the flow file
    :return: optical flow data in matrix

    adapted from https://github.com/liruoteng/OpticalFlowToolkit/

    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        raise TypeError('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        if verbose:
            print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=int(2 * w * h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


def draw_hsv(flow, scale):
    """
    Draw optical flow data (Middlebury format)
    :param flow: optical flow data in matrix
    :return: scale: scale for representing the optical flow
    adapted from https://github.com/npinto/opencv/blob/master/samples/python2/opt_flow.py
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * scale, 255)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    return rgb


def generate_wheel(size):
    """
     Generate wheel optical flow for visualizing colors
     :param size: size of the image
     :return: flow: optical flow for visualizing colors
     """
    rMax = size / 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    u = x - size / 2
    v = y - size / 2
    r = np.sqrt(u ** 2 + v ** 2)
    u[r > rMax] = 0
    v[r > rMax] = 0
    flow = np.dstack((u, v))

    return flow


def normalized_cross_correlation(patch: np.array, search_area: np.array) -> np.array:
    """
    Estimate normalized cross correlation values for a patch in a searching area.
    """
    # Complete the function
    i0 = patch
    i0_mean = np.mean(i0)
    # ....
    result = np.zeros(search_area.shape, dtype=float)
    margin_y = int(patch.shape[0]/2)
    margin_x = int(patch.shape[1]/2)

    for i in range(margin_y, search_area.shape[0] - margin_y):
        for j in range(margin_x, search_area.shape[1] - margin_x):
            i1 = search_area[i-margin_x:i + margin_x + 1, j-margin_y:j + margin_y + 1]
            i1_mean = np.mean(i1)
            # Implement the correlation
            result[i, j] = np.sum((i0 - i0_mean) * (i1 - i1_mean)) / (np.sqrt(np.sum((i0 - i0_mean) ** 2)) * np.sqrt(np.sum((i1 - i1_mean) ** 2)))

    return result


def seed_estimation_NCC_single_point(img1_gray, img2_gray, i_img, j_img, patch_half_size: int = 5, searching_area_size: int = 100):

    # Attention!! we are not checking the padding
    patch = img1_gray[i_img - patch_half_size:i_img + patch_half_size + 1, j_img - patch_half_size:j_img + patch_half_size + 1]

    i_ini_sa = i_img - int(searching_area_size / 2)
    i_end_sa = i_img + int(searching_area_size / 2) + 1
    j_ini_sa = j_img - int(searching_area_size / 2)
    j_end_sa = j_img + int(searching_area_size / 2) + 1

    search_area = img2_gray[i_ini_sa:i_end_sa, j_ini_sa:j_end_sa]
    result = normalized_cross_correlation(patch, search_area)

    iMax, jMax = np.where(result == np.amax(result))

    i_flow = i_ini_sa + iMax[0] - i_img
    j_flow = j_ini_sa + jMax[0] - j_img

    return i_flow, j_flow

def numerical_gradient(img_int: np.array, point: np.array)->np.array:
    """
    https://es.wikipedia.org/wiki/Interpolaci%C3%B3n_bilineal
    :param img:image to interpolate
    :param point: [[y0,x0],[y1,x1], ... [yn,xn]]
    :return: Ix_y = [[Ix_0,Iy_0],[Ix_1,Iy_1], ... [Ix_n,Iy_n]]
    """

    a = np.zeros((point.shape[0], 2), dtype= float)
    filter = np.array([-1, 0, 1], dtype=float)
    point_int = point.astype(int)
    img = img_int.astype(float)

    for i in range(0,point.shape[0]):
        py = img[point_int[i,0]-1:point_int[i,0]+2,point_int[i,1]].astype(float)
        px = img[point_int[i,0],point_int[i,1]-1:point_int[i,1]+2].astype(float)
        a[i, 0] = 1/2*np.dot(filter,px)
        a[i, 1] = 1/2*np.dot(filter,py)

    return a

def int_bilineal(img: np.array, point: np.array)->np.array:
    """
    https://es.wikipedia.org/wiki/Interpolaci%C3%B3n_bilineal
    Vq = scipy.ndimage.map_coordinates(img.astype(np.float), [point[:, 0].ravel(), point[:, 1].ravel()], order=1, mode='nearest').reshape((point.shape[0],))

    :param img:image to interpolate
    :param point: point subpixel
    point = [[y0,x0],[y1,x1], ... [yn,xn]]
    :return: [gray0,gray1, .... grayn]
    """
    A = np.zeros((point.shape[0], 2, 2), dtype=float)
    point_lu = point.astype(int)
    point_ru = np.copy(point_lu)
    point_ru[:,1] = point_ru[:,1] + 1
    point_ld = np.copy(point_lu)
    point_ld[:, 0] = point_ld[:, 0] + 1
    point_rd = np.copy(point_lu)
    point_rd[:, 0] = point_rd[:, 0] + 1
    point_rd[:, 1] = point_rd[:, 1] + 1

    A[:, 0, 0] = img[point_lu[:,0],point_lu[:,1]]
    A[:, 0, 1] = img[point_ru[:,0],point_ru[:,1]]
    A[:, 1, 0] = img[point_ld[:,0],point_ld[:,1]]
    A[:, 1, 1] = img[point_rd[:,0],point_rd[:,1]]
    l_u = np.zeros((point.shape[0],1,2),dtype= float)
    l_u[:, 0, 0] = -((point[:,0]-point_lu[:,0])-1)
    l_u[:, 0, 1] = point[:,0]-point_lu[:,0]

    r_u = np.zeros((point.shape[0],2,1),dtype= float)
    r_u[:, 0, 0] = -((point[:,1]-point_lu[:,1])-1)
    r_u[:, 1, 0] = point[:, 1]-point_lu[:,1]
    grays = l_u @ A @ r_u

    return grays.reshape((point.shape[0],))

def lucas_kanade_refinement(img1_gray, img2_gray, points_selected, seed_optical_flow_sparse, patch_half_size):
    # Initialize the optical flow vectors with the sparse optical flow estimation
    optical_flow = seed_optical_flow_sparse.copy()

    for k in range(0, points_selected.shape[0]):
        print("Processing point " + str(k) + " of " + str(points_selected.shape[0]))
        # Extract the patch around the point in the first image
        i, j = points_selected[k,1], points_selected[k,0]
        coord_patch0 = np.zeros((patch_half_size * 2 + 1, patch_half_size * 2 + 1, 2))
        for i_patch in range(-patch_half_size, patch_half_size + 1):
            for j_patch in range(-patch_half_size, patch_half_size + 1):
                coord_patch0[i_patch + patch_half_size, j_patch + patch_half_size, :] = np.array([i + i_patch, j + j_patch])
        # coord_patch0 = coord_patch0.astype(int)
        coord_patch0 = coord_patch0.reshape((coord_patch0.shape[0] * coord_patch0.shape[1], coord_patch0.shape[2]))
        patch0 = int_bilineal(img1_gray, coord_patch0)
        print(patch0.shape)
        # print(patch0)
        # patch0 = img1_gray[i - patch_half_size:i + patch_half_size + 1, j - patch_half_size:j + patch_half_size + 1]

        # Compute the image gradients within the patch
        # grad_i, grad_j = np.gradient(patch0)
        gradient = numerical_gradient(img1_gray, coord_patch0)
        print(gradient.shape)
        grad_i = gradient[:, 1]
        grad_j = gradient[:, 0]

        # Compute the Jacobian matrix from the image gradients
        J = np.array([grad_i.flatten(), grad_j.flatten()]).T
        print(J.shape)

        # Compute matrix A from the Jacobian matrix
        A = np.array([[np.sum(grad_i ** 2), np.sum(grad_i * grad_j)], [np.sum(grad_i * grad_j), np.sum(grad_j ** 2)]])
        print(A.shape)

        # Check if A is invertible calculating its determinant
        detA = np.linalg.det(A)
        if detA < 1e-6:
            print("A is not invertible")
            continue
        
        print(A)

        epsilon = 1e-6
        delta_u = np.ones(optical_flow.shape[1])
        u = optical_flow[k, :]
        # Iterate until convergence
        while np.sqrt(np.sum(delta_u ** 2)) > epsilon:

            # Compute the patch in the second image
            # u = optical_flow[k, :]
            coord_patch1 = coord_patch0 + u
            patch1 = int_bilineal(img2_gray, coord_patch1)
            # patch1 = img2_gray[i - patch_half_size + int(u[0]):i + patch_half_size + 1 + int(u[0]), j - patch_half_size + int(u[1]):j + patch_half_size + 1 + int(u[1])]

            # Compute the error between the two patches
            error = patch1 - patch0

            # Compute the optical flow increment
            b = np.array([-np.sum(grad_i * error), -np.sum(grad_j * error)])

            # Solve the linear system
            delta_u = np.linalg.solve(A, b)

            # Update the optical flow
            u = u + delta_u
            print(u)
        optical_flow[k, :] = np.array([u[1], u[0]])
            
    return optical_flow

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)
    unknownFlowThresh = 1e9

    flow_12 = read_flo_file("flow10.flo", verbose=True)
    binUnknownFlow = flow_12 > unknownFlowThresh

    # Adding random noise to the gt optical flow for plotting example
    flow_est = flow_12 * np.bitwise_not(binUnknownFlow) + np.random.rand(flow_12.shape[0], flow_12.shape[1], flow_12.shape[2]) * 1.2 - 0.6

    img1 = read_image("frame10.png")
    img2 = read_image("frame11.png")

    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # List of sparse points
    points_selected = np.loadtxt('points_selected.txt')
    points_selected = points_selected.astype(int)

    template_size_half = 5 # for a 11x11 template
    searching_area_size: int = 15

    seed_optical_flow_sparse = np.zeros((points_selected.shape))
    for k in range(0,points_selected.shape[0]):
        i_flow, j_flow = seed_estimation_NCC_single_point(img1_gray, img2_gray, points_selected[k,1], points_selected[k,0], template_size_half, searching_area_size)
        seed_optical_flow_sparse[k,:] = np.hstack((j_flow,i_flow))

    print(seed_optical_flow_sparse)

    # Refine with Lucas Kanade
    optical_flow = lucas_kanade_refinement(img1_gray, img2_gray, points_selected, seed_optical_flow_sparse, template_size_half)

    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)
    flow_error = optical_flow - flow_gt
    # flow_error[binUnknownFlow] = 0
    error_norm = np.sqrt(np.sum(flow_error ** 2, axis=-1))

    print(flow_gt)
    print(optical_flow)
    flow_est_sparse = optical_flow
    flow_est_sparse_norm = np.sqrt(np.sum(flow_est_sparse ** 2, axis=1))
    error_sparse = flow_est_sparse - flow_gt
    error_sparse_norm = np.sqrt(np.sum(error_sparse ** 2, axis=1))

    # Plot results for sparse optical flow
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img1)
    axs[0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_est_sparse_norm[k]), color='r')
    axs[0].quiver(points_selected[:, 0], points_selected[:, 1], flow_est_sparse[:, 0], flow_est_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[0].title.set_text('Optical flow')
    axs[1].imshow(img1)
    axs[1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_sparse_norm[k]),
                    color='r')
    axs[1].quiver(points_selected[:, 0], points_selected[:, 1], error_sparse[:, 0], error_sparse[:, 1], color='b',
               angles='xy', scale_units='xy', scale=0.05)

    axs[1].title.set_text('Error with respect to GT')
    plt.show()    
