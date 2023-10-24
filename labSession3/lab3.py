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

def calculate_RANSAC_own(gray, gray2):
    kp1, desc1 = SIFT_keypoints(gray,1000)
    kp2, desc2  = SIFT_keypoints(gray2,1000)
    distRatio = 0.99
    minDist = 500
    matches1 = matchWith2NDRR(desc1, desc2, distRatio, minDist)

    num_samples = 4
    best_model = None
    finished = False
    añadir = True

    
    while not finished:

        np.random.shuffle(matches1)
        matches = matches1[:num_samples]


        src_pts = np.float32([ kp1[m[0]].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[1]].pt for m in matches ]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, 0,5.0)
        if H is not None:
            
        
            matchesMask = mask.ravel().tolist()
    
            rest_matches = matches1[num_samples:]
            model = 0
            if H is not None:
                for m in rest_matches:
                    pts = np.float32( kp1[m[0]].pt ).reshape(-1,1,2)
                    dst2 = cv2.perspectiveTransform(pts,H)
                    dst = dst2[0][0]
                    x,y = kp2[m[1]].pt
                    
                    err = math.sqrt((dst[0] - x) ** 2 + (dst[1] - y) ** 2)
                    if err < 2:
                       model += 1 
                if model >= 20:
                    best_model = H
                    finished = True
            
    
    return best_model, añadir


if __name__ == '__main__':
    
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

    # Conversion from DMatches to Python list
    matchesList = matchesListToIndexMatrix(dMatchesList)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    # Matched points in homogeneous coordinates
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

    """ It is quite difficult to reduce the number of false positive matches because some features 
        found in the image one are not included in the image two. So, all the mathes given by SIFT
        on those features are false positives.
    """

    # PART 2
    print("PART 2")
    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    # Select a threshold to have a low false positive rate
    distRatio = 0.99
    minDist = 500
    matchesList = matchWith2NDRR_2(descriptors_1, descriptors_2, distRatio, minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    
    # Plot the first 10 matches
    imgMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.waitforbuttonpress()

    # Conversion from DMatches to Python list
    matchesList = matchesListToIndexMatrix(dMatchesList)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    # Matched points in homogeneous coordinates
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))

    # PART 3

    # path = './output/image1_image2_matches.npz'
    # npz = np.load(path)
    #npz.files
    #npz['matches']

    # PART 4
    print("PART 4")
    H, good_model = calculate_RANSAC_own(image_pers_1, image_pers_2)
    print(H, " ", good_model)