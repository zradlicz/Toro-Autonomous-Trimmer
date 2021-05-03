# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:36:24 2021

@author: Alan

from: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
"""

import numpy as np
import cv2 as cv
import glob


# resulting matrices from camera calibration code in __main__
mtx = np.array([[892.99442461,   0.        , 637.77074406],
       [  0.        , 892.63477064, 370.8337794 ],
       [  0.        ,   0.        ,   1.        ]])

newcameramtx = np.array([[890.89001465,   0.        , 637.77908467],
       [  0.        , 891.75317383, 372.43966263],
       [  0.        ,   0.        ,   1.        ]])

dist = np.array([[ 0.18370311, -0.54124848,  0.00177504,  0.00000163,  0.39589324]])


def undistort_color(img):
    return cv.undistort(img, mtx, dist, None, newcameramtx)

if __name__ == "__main__":
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    gray=[]
    images = glob.glob('camera calibration data/color*.png')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (9,6), corners2, ret)
            cv.imshow('img', img)
            while 1:
                key = cv.waitKey(1)
                if key & 0xFF == ord('y'):
                    key = 0
                    break
    cv.destroyAllWindows()
    
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    
    print("Distortion:")
    print(dist)
    
    img = cv.imread('camera calibration data/color3.png')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    cv.imwrite('camera calibration data/before.png',img)
    cv.imwrite('camera calibration data/after.png',dst)
    
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )