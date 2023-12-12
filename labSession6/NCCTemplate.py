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


def read_image(filename: str, ):
    """
    Read image using opencv converting from BGR to RGB
    :param filename: name of the image
    :return: np matrix with the image
    """
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

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

def lucas_kanade_refinement(img1_gray, img2_gray, points_selected, seed_optical_flow_sparse, patch_half_size):
    # Initialize the optical flow vectors with the sparse optical flow estimation
    optical_flow = seed_optical_flow_sparse.copy()

    for k in range(points_selected.shape[0]):
        # Extract the patch around the point in the first image
        i, j = points_selected[k,1], points_selected[k,0]
        patch0 = img1_gray[i - patch_half_size:i + patch_half_size + 1, j - patch_half_size:j + patch_half_size + 1]

        # Compute the image gradients within the patch
        grad_i, grad_j = np.gradient(patch0)

        # Compute the Jacobian matrix from the image gradients
        J = np.array([grad_j.flatten(), grad_i.flatten()]).T

        # Compute matrix A from the Jacobian matrix
        A = J.T @ J

        # Check if A is invertible calculating its determinant
        detA = np.linalg.det(A)
        if detA < 1e-6:
            print("A is not invertible")
            continue
        
        print(A)

        # Iterate until convergence
        while True:

            # Compute the patch in the second image
            i_flow, j_flow = optical_flow[k, :]
            i_flow = int(i_flow)
            j_flow = int(j_flow)
            patch1 = img2_gray[i - patch_half_size + i_flow:i + patch_half_size + 1 + i_flow, j - patch_half_size + j_flow:j + patch_half_size + 1 + j_flow]
            
            # Compute the error between the two patches
            error = patch1 - patch0

            # Compute the optical flow increment
            b = J.T @ error.flatten()

            # Solve the linear system
            delta_optical_flow = np.linalg.inv(A) @ b

            # Update the optical flow
            optical_flow[k, :] += delta_optical_flow
            
    return optical_flow

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

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
    
        
