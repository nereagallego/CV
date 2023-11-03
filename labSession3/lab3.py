# Computer Vision Laboratory 3
# Authors: César Borja and Nerea Gallego

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import math

def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList


def matchWith2NDRR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        if (dist[indexSort[0]] < minDist):
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches

def matchWith2NDRR_2(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        # Find the second closest match
        if (dist[indexSort[1]]*distRatio < dist[indexSort[0]]):
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches

def SIFT_keypoints(gray, nfeatures : int, contrastThreshold=0.04, sigma=1.6):

    sift = cv2.SIFT_create(nfeatures)
    kp, desc = sift.detectAndCompute(gray,None)

    return kp, desc

def computeHomography(points1, points2):
    # print(points1.shape)
    A = np.zeros((points1.shape[1] * 2, 9))
    for i in range(points1.shape[1]):
        # A[2*i, :] = [points1[i,0], points1[i,1], 1.0, 0.0, 0.0, 0.0, -points2[i,0]*points1[i,0], -points2[i,0]*points1[i,1], -points2[i,0]]
        # A[2*i+1,:] = [0.0, 0.0, 0.0, points1[i,0], points1[i,0], 1.0, -points2[i,1]*points1[i,0], -points2[i,1]*points1[i,1], -points2[i,1]]
        # A[2*i, :] = [points1[0,i], points1[1,i], 1.0, 0.0, 0.0, 0.0, -points2[0,i]*points1[0,i], -points2[0,i]*points1[1,i], -points2[0,i]]
        # A[2*i+1,:] = [0.0, 0.0, 0.0, points1[0,i], points1[0,i], 1.0, -points2[1,i]*points1[0,i], -points2[1,i]*points1[1,i], -points2[1,i]]
        A[2*i, :] = [points1[0,i], points1[1,i], 1.0, 0.0, 0.0, 0.0, -points2[0,i]*points1[0,i], -points2[0,i]*points1[1,i], -points2[0,i]]
        A[2*i+1,:] = [0.0, 0.0, 0.0, points1[0,i], points1[1,i], 1.0, -points2[1,i]*points1[1,i], -points2[1,i]*points1[1,i], -points2[1,i]]
    
    
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H/H[2,2]

def calculate_RANSAC_own_H(source,dst,threshold):
    num_samples = 4
    num_Attempts = 5000

    matches = np.vstack((source,dst))
    best_model_votes = 0
    best_model_matches = None

    for kAttempt in range(num_Attempts):
        votes = 0

        rng = np.random.default_rng()
        indx_subset = rng.choice(matches.shape[1] - 1, size=num_samples, replace=False)
        matches_subset = []
        rest_matches = []

        for i in range(matches.shape[1]):
            if i in indx_subset:
                matches_subset.append(matches[:, i])
            else:
                rest_matches.append(matches[:, i])

        matches_subset = np.array(matches_subset).T
        rest_matches = np.array(rest_matches).T

        H = computeHomography(matches_subset[0:3,:],matches_subset[3:6,:])

        for i in range(rest_matches.shape[1]):

            x1 = rest_matches[0:3, i]
            x2 = rest_matches[3:6, i]

            pred = H @ x1
            pred = pred / pred[2]


            # error the Euclidian distance L2 between the matched point and the transformed point from the other image
            dist = np.abs( np.linalg.norm(x2 - pred) )

            if dist < threshold:
                votes = votes + 1

        if votes > best_model_votes:
            best_model_votes = votes
            print(votes)
            H_most_voted = H
            best_model_matches = matches_subset

    return H_most_voted, best_model_matches

def compute_epipolar_line(x1, F):
    # Convert clicked point to homogeneous coordinates
    x1 = np.append(x1, 1)

    # Compute the epipolar line
    l = np.dot(F, x1)

    # Normalize the line
    l = l / np.linalg.norm(l)

    return l

# def get_normalization_transform(point_set):
#     point_set = np.array(point_set)
#     x = point_set[:, 0]
#     y = point_set[:, 1]
    
#     centroid_x = np.mean(x)
#     centroid_y = np.mean(y)
#     x_new = x - centroid_x
#     y_new = y - centroid_y

#     mean_distance = np.mean(np.sqrt(x_new**2 + y_new**2))
#     scale = np.sqrt(2)/mean_distance
    
#     T = np.eye(3)
#     T[0, 0] = scale
#     T[0, 2] = -scale*centroid_x
#     T[1, 1] = scale
#     T[1, 2] = -scale*centroid_y
    
#     return T

def compute_fundamental_matrix(points1, points2):
    # Normalize the points
    # T1 = get_normalization_transform(points1.T)
    # T2 = get_normalization_transform(points2.T)
    # points1 = np.dot(T1, points1)
    # points2 = np.dot(T2, points2)
    # Compute the fundamental matrix
    # print(points1.shape[0], " ", points1.shape[1])
    A = np.zeros((points1.shape[1], 9))
    for i in range(points1.shape[1]):
        A[i, :] = [points1[0, i] * points2[0, i], points2[0, i] * points1[1, i], points2[0, i], points1[0, i] * points2[1, i], points1[1, i] * points2[1, i], points2[1,i], points1[0,i], points1[1,i], 1]
        # A[i, :] = [points1[i,0] * points2[i,0], points2[i,0] * points1[i,0], points2[i,0], points1[i,0] * points2[i, 1], points1[i, 1] * points2[i, 1], points2[i,1], points1[i,0], points1[i,1], 1]
    
    _, _, V = np.linalg.svd(A)

    # compute the fundamental matrix from the right singular vector corresponding to the smallest singular value
    F = V[-1, :].reshape((3, 3))
    U, S, V = np.linalg.svd(F)

    # enforce rank 2 constraint
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F/F[2,2]

def calculate_RANSAC_own_F(source,dst,threshold):
    num_samples = 8
    spFrac = 0.5  # spurious fraction
    P = 0.999  # probability of selecting at least one sample without spurious

    # number m of random samples
    nAttempts = np.round(np.log(1 - P) / np.log(1 - np.power((1 - spFrac),num_samples)))
    num_attempts = nAttempts.astype(int)
    num_attempts = 5000

    matches = np.vstack((source,dst))
    best_model_votes = 0
    best_model_matches = None

    for kAttempt in range(num_attempts):
        votes = 0

        rng = np.random.default_rng()
        indx_subset = rng.choice(matches.shape[1] - 1, size=num_samples, replace=False)
        matches_subset = []
        rest_matches = []

        for i in range(matches.shape[1]):
            if i in indx_subset:
                matches_subset.append(matches[:, i])
            else:
                rest_matches.append(matches[:, i])

        matches_subset = np.array(matches_subset).T
        rest_matches = np.array(rest_matches).T

        # F = find_fundamental_matrix(matches_subset[0:3,:],matches_subset[3:6,:], nx1, ny1, nx2, ny2)
        F = compute_fundamental_matrix(matches_subset[0:3,:],matches_subset[3:6,:])
        if F is not None:
            for i in range(rest_matches.shape[1]):

                x1 = rest_matches[0:3, i]
                x2 = rest_matches[3:6, i]

                l_2 = F @ x1
                    
                dist_x2_l2 = np.abs(np.dot(x2.T,np.dot(F , x1))/ np.sqrt((l_2[0]**2 + l_2[1]**2)))

                if dist_x2_l2 < threshold:
                    votes = votes + 1

            if votes > best_model_votes:
                best_model_votes = votes
                print(votes)
                F_most_voted = F
                best_model_matches = matches_subset

    return F_most_voted, best_model_matches

# Compute the epipole from the fundamental matrix
def compute_epipole(F):
    # Compute the epipole
    _, _, V = np.linalg.svd(F)
    e = V[-1, :]
    e = e / e[-1]
    return e

def show_epipolar_lines(img1, img2, F):
    
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

def normalizationMatrix(nx,ny):
    """
    Estimation of fundamental matrix(F) by matched n matched points.
    n >= 8 to assure the algorithm.

    -input:
        nx: number of columns of the matrix
        ny: number of rows of the matrix
    -output:
        Nv: normalization matrix such that xN = Nv @ x
    """
    Nv = np.array([[1/nx, 0, -1/2], [0, 1/ny, -1/2], [0, 0, 1]])
    return Nv


if __name__ == '__main__':
    color = ['+b', '+g', '+c', '+m', '+y', '+k', '+w', '+p' ]

    # PART 1
    print("PART 1")
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Images path
    timestamp1 = '1403715282262142976'
    timestamp2 = '1403715413262142976'

    path_image_1 = 'image1.png'
    path_image_2 = 'image2.png'

    # Read images
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    # Select a threshold to have a low false positive rate
    distRatio = 0.80
    minDist = 75
    matchesList = matchWith2NDRR(descriptors_1, descriptors_2, distRatio, minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    
    # Plot the first 10 matches
    imgMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()

    """ It is quite difficult to reduce the number of false positive matches because some features 
        found in the image one are not included in the image two. So, all the mathes given by SIFT
        on those features are false positives.
    """

    # PART 2
    print("PART 2")
    
    # # Select a threshold to have a low false positive rate
    # distRatio = 0.99
    # minDist = 500
    # matchesList = matchWith2NDRR_2(descriptors_1, descriptors_2, distRatio, minDist)
    # dMatchesList = indexMatrixToMatchesList(matchesList)
    # dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    
    # # Plot the first 10 matches
    # imgMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
    #                              None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    # plt.draw()
    # plt.waitforbuttonpress()
    # plt.close()

    # PART 3

    # path = './output/image1_image2_matches.npz'
    # npz = np.load(path)
    #npz.files
    #npz['matches']

    # PART 4
    print("PART 4")

     # Select a threshold to have a low false positive rate
    distRatio = 0.99
    minDist = 75
    matchesList = matchWith2NDRR_2(descriptors_1, descriptors_2, distRatio, minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    # Conversion from DMatches to Python list
    matchesList = matchesListToIndexMatrix(dMatchesList)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    # Matched points in homogeneous coordinates
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

    # H, x = calculate_RANSAC_own_H(x1, x2, 5)
    # print(H)

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # plt.suptitle('Homography')
    # ax[0].imshow(image_pers_1)
    # ax[0].set_title('Image 1')
    # ax[1].imshow(image_pers_2)
    # ax[1].set_title('Image 2')

    # for i in range(4):
    #     # plt.subplot(ax[0])
    #     ax[0].plot(x[0, i], x[1, i], '+r')
    #     ax[1].plot(x[3, i], x[4, i], '+r')
    #     plt.draw()
    # plt.waitforbuttonpress()

    # plt.suptitle('Click points in image 1 to transform to image 2')
    # for i in range(4):
    #     coord_click = plt.ginput(1, show_clicks=False)
    #     coor = np.array([coord_click[0][0], coord_click[0][1], 1.0])
    #     ax[0].plot(coor[0], coor[1], color[i], markersize=10)
    #     plt.draw()
    #     coord_hom = H @ coor
    #     coord_hom = coord_hom / coord_hom[2]
    #     ax[1].plot(coord_hom[0], coord_hom[1], color[i], markersize=10)
    #     plt.draw()

    # plt.waitforbuttonpress()

    # plt.close()

    # PART 5
    print("PART 5")
    
    F, x = calculate_RANSAC_own_F(x1, x2, 2)
    # F, x = calculate_RANSAC_own_F(x1, x2, 5)
    print(F)

    # fig2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
    # plt.suptitle("Points to calculate F")
    # ax2[0].set_title('Image 1')
    # ax2[0].imshow(image_pers_1)
    # ax2[1].set_title('Image 2')
    # ax2[1].imshow(image_pers_2)

    # print the epipolar lines when clicking on the images
    show_epipolar_lines(image_pers_1, image_pers_2, F)
   

    # plt.close()