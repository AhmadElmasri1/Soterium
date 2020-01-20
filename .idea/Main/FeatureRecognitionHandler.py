import cv2
import numpy as np 
import sys
from matplotlib import pyplot as plt


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

