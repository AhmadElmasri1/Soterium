import cv2
import numpy as np 
import sys
from matplotlib import pyplot as plt
import venv


#Corner detection algorithm using the harris method
def CornerHarris(inputImage):

    #Generate a grayscale image in order to simplify corner detection
    grayImg = cv2.cvtColor(inputImage.copy(), cv2.COLOR_BGR2GRAY)
    grayImg = np.float32(grayImg)
    dstImg = cv2.cornerHarris(grayImg, 2, 23, 0.04)

    imgHolder = inputImage.copy()
    imgHolder[dstImg>0.01 * dstImg.max()] = [0,0,255]

    return imgHolder

def FeatureSift(inputImage):

    grayImg = cv2.cvtColor(inputImage.copy(), cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(grayImg, None)

    dstImg = inputImage.copy()

    imgHolder = cv2.drawKeypoints(inputImage, keypoints, dstImg,
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = (51,263, 2356))

    return dstImg

def FeatureSurf(inputImage, threshold = 4000):
    grayImg = cv2.cvtColor(inputImage.copy(), cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create(threshold)
    keypoints, descriptor = surf.detectAndCompute(grayImg, None)

    dstImg = grayImg.copy()

    dstImg = cv2.drawKeypoints(inputImage, keypoints, dstImg,
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color = (51,263, 2356))

    return dstImg

def FeatureMatchingKNN(inputImage, referenceImage):
    orb = cv2.ORB_create()
    grayImageIn = cv2.cvtColor(inputImage.copy(), cv2.COLOR_BGR2GRAY)
    grayImageRef = cv2.cvtColor(referenceImage.copy(), cv2.COLOR_BGR2GRAY)
    # cv2.color

    keypoint1, descriptor1 = orb.detectAndCompute(grayImageIn, None)
    keypoint2, descriptor2 = orb.detectAndCompute(grayImageRef, None)

    bruteForce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bruteForce.knnMatch(descriptor1,descriptor2,k=1)

    outputImage = inputImage.copy()

    holderImg = cv2.drawMatchesKnn(grayImageIn, keypoint1, grayImageRef, keypoint2,
    matches, grayImageIn, flags = 2)

    plt.imshow(holderImg)
    plt.show()

    return inputImage.copy()

def FeatureMatchingFLANN(inputImage, referenceImage):

    sift = cv2.xfeatures2d.SIFT_create()

    grayImageIn = cv2.cvtColor(inputImage.copy(), cv2.COLOR_BGR2GRAY)
    grayImageRef = cv2.cvtColor(referenceImage.copy(), cv2.COLOR_BGR2GRAY)

    keypoints1, descriptors1 = sift.detectAndCompute(grayImageIn, None)
    keypoints2, descriptors2 = sift.detectAndCompute(grayImageRef, None)

    FLANN_INDEX_KDTREE = 0
    indexParameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    searchParameters= dict(checks = 50)

    flann = cv2.FlannBasedMatcher(indexParameters, searchParameters)

    matches = flann.knnMatch(descriptors1, descriptors2, k = 2)

    matchesMask = [[0,0] for i in range(len(matches))]#len(matches)

    # David G. Lowe's ratio test, populate the mask
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    drawParameters = dict(matchColor = (0,255,0),
                      singlePointColor = (255,0,0),
                      matchesMask = matchesMask,
                      flags = 0)

    resultImage = cv2.drawMatchesKnn(inputImage, keypoints1, referenceImage
                                     ,keypoints2, matches, None, **drawParameters)
    plt.imshow(resultImage)
    plt.show()

def FeatureMatchingFLANNH(inputImage, referenceImage):
    MIN_MATCH_COUNT = 10

    img1 = cv2.cvtColor(inputImage.copy(), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(referenceImage.copy(), cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good
                               ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good
                               ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" %(len(good),MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()

