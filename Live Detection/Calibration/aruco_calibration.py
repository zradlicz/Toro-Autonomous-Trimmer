# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:07:46 2021

@author: Alan

following https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/
https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html

camera calibration info
http://ksimek.github.io/2013/08/13/intrinsic/
https://learnopencv.com/understanding-lens-distortion/
http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf

Modify any settings in the Calibration settings section, then run this file to
calculate a calibration between image pixels and gantry coordinates. Plots 
generated are for the calibration that was just calculated.

The calibration data collected using calibrate_trimmer.py should be placed in a 
folder in the same location as this file. The file structure should look like:

Calibration/
    aruco_calibration.py
    calibration7/
        color0.png
        color1.png
        ...
        depth_raw0.npz
        depth_raw1.npz
        ...
        depth0.npz
        depth1.npz
        ...
        gantry_coord.txt

Import this file and call pixel_to_gantry() to use the calibration. This function
uses the globally defined calibration matrices tf_c2r and error_coeff. 
"""

import os

import pyrealsense2 as rs
import cv2
from cv2 import aruco
import numpy as np
np.set_printoptions(suppress = 1)
import matplotlib.pyplot as plt


"""
##############################################################################
Global variables
##############################################################################
"""
"""
# Marker generation ##########################################################
"""
# pixels per inch, for if you want to print a marker with a precise size
PRINT_PPI = 100

# smallest aruco dictionary chosen because we don't need that many markers, 
# and because smaller markers = less bits in the marker = easier to detect
dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)

markers = [2,16,24,33,39,45] # ids of the aruco markers to generate

marker_side_length = 5 # I'm not sure this even matters

# sets up an instance of a charuco marker use for both generation and detection
chboard = aruco.CharucoBoard_create(3,3,100,80,dictionary)

"""
# Marker detection ###########################################################
"""
# camera calibration coefficients for our intel realsense
matrix_coefficients = np.array([[916.05,0,637.569], # for RGB camera
                                [0,916.009,363.627],
                                [0,0,1]])
'''
matrix_coefficients = np.array([[426.869,0,420.241], # for IR camera
                                [0,426.869,239.53],
                                [0,0,1]])'''
distortion_coefficients = np.array([0,0,0,0.0])

# the same camera calibration coefficients but in a form that works with the
# pyrealsense2 module
color_intr = rs.pyrealsense2.intrinsics()
attr = {'coeffs':[0.0, 0.0, 0.0, 0.0, 0.0],
 'fx':916.0498046875,
 'fy':916.0092163085938,
 'height':720,
 'model':rs.pyrealsense2.distortion.inverse_brown_conrady,
 'ppx':637.5685424804688,
 'ppy':363.62677001953125,
 'width':1280}
for k in attr:
    color_intr.__setattr__(k, attr[k])

# depth scale for the realsense camera
# this is depth_sensor.get_depth_scale() times 1000 so that it is in mm instead of m
depth_scale = 1.0000000474974513

"""
# Marker calibration #########################################################
"""
# controls whether to undistort camera images using the calibration method
# provided by opencv
# this doesn't work that well and doesn't result in significantly less calibration
# error, but does sometimes result in a better-looking error distribution
UNDISTORT = False

if UNDISTORT:
    import opencv_camera_calibration

# these two arrays are the ones used in pixel_to_gantry(). If a new calibration is
# generated using this script, these two arrays need to be manually changed in
# order to use that new calibration.
# This specific calibration was generated using the settings below:
# CALIB_FOLDER = "calibration7"
# CALIB_POINT_INDEX = [0,20,75,4]
# SKIP_POINTS = [241,234,244]
# UNDISTORT = False

tf_c2r = np.array([[-0.9634852479151038,
  -0.019112183589649483,
  0.031138259013934882,
  65.92041806485025],
 [0.012051767154691247,
  -0.9573818243636759,
  0.02134937764826797,
  263.5963078893157],
 [-0.01306539421904099,
  -0.006062934787798813,
  -0.9085368631585232,
  720.3752560854116],
 [0.0, 0.0, 0.0, 1.0]])

error_coeff = np.array([[2.5798425474065166e-05, 6.41249514546737e-05, 9.798959549221503e-06],
 [-5.53819948143682e-06, 3.5583110726905504e-05, 0.0001554245522341734],
 [-7.092627305252204e-05, -0.0002331316184827212, -0.00027929848106944647],
 [1.002563970887976, -0.007089276810677255, 0.0016043302675154342],
 [0.004956270054719525, 0.9875332619482434, -0.06701689262215534],
 [0.012159272729270596, 0.038843468814601496, 1.0600331571437522],
 [-0.6836834441814134, 0.5643210052063452, 3.222290758916191]])

"""
# Calibration settings #######################################################
"""
# these settings only affect the code in __main__ and so do nothing when this
# file is imported as a module

# NOTE: two other options that are not easily changed using a global variable 
# are described below:
#
# the option to use a planar fit to calculate the marker center depth:
# This can be changed by modifying extract_charuco_depth().
#
# whether to use depth_rawXX.npz or depthXX.npz:
# depthXX.npz are the depth_raw images after being passed through the filters 
# defined in camera.py. This can be changed by modifying load_images().

CALIB_FOLDER = "calibration7"

# indicies of the points to use for origin, x-axis, y-axis, and z-axis
# indicies correspond to the points listed in CALIB_FOLDER/gantry_coord.txt
#CALIB_POINT_INDEX = [1,3,5,2] # calibration1
#CALIB_POINT_INDEX = [0,1,3,2] # calibration3
#CALIB_POINT_INDEX = [0,9,30,2] # calibration4
#CALIB_POINT_INDEX = [0,12,75,2] # calibration4
#CALIB_POINT_INDEX = [0,3,15,1] # calibration4
#CALIB_POINT_INDEX = [37,40,52,38] # calibration4
#CALIB_POINT_INDEX = [0,45,850,4] # calibration5
#CALIB_POINT_INDEX = [0,5,25,2] # calibration6
CALIB_POINT_INDEX = [0,20,75,4] # calibration7

# some points may be outliers or have blurry markers, so skip these
SKIP_POINTS = []
#SKIP_POINTS = [340,668,763,135,805,830] # calibration5
#SKIP_POINTS = [38] # calibration6
#SKIP_POINTS = [38,39,5,97] # calibration6
SKIP_POINTS = [241,234,244] # calibration7

# if RECALCULATE is set True, this script will redetect all of the markers and
# recalculate their locations in the camera coordinate system
RECALCULATE = False

"""
##############################################################################
Marker generation
##############################################################################

To generate a marker like ones used in the current calibration, run:
generate_charuco(2,2)
"""

# This creates individual aruco markers, which were replaced with charuco markers
# for the final calibration. This code is left here in case it may be useful.
def generate_markers(marker_size,border_w,border_b):
    for markid in markers:
        markerImage = np.zeros((marker_size, marker_size), dtype=np.uint8)
        markerImage = aruco.drawMarker(dictionary, markid, marker_size, markerImage, 1)
                
        markerImage = np.pad(markerImage,[(border_w,border_w),(border_w,border_w)],constant_values=255)
        markerImage = np.pad(markerImage,[(border_w,border_w),(border_w,border_w)],constant_values=0)
        
        cv2.imwrite("aruco4x4_50_marker%s.png"%markid,markerImage)
        
        plt.imshow(markerImage)
        
        plt.show()

        return markerImage

# This creates a printable image of a charuco marker
def generate_charuco(width_in,height_in):
    img = chboard.draw((PRINT_PPI*width_in,PRINT_PPI*height_in))
    plt.imshow(img)
    cv2.imwrite("charuco4x4_50_marker%sby%s.png"%(width_in,height_in),img)

# pads arrays with a constant value, just make sure to give it a shape bigger
# than that of arr
def pad_both_to(arr,shape,constant_values=0):
    padding = []
    for o_s,n_s in zip(arr.shape,shape):
        Lpadding = int((n_s-o_s)/2)
        Rpadding = n_s-o_s - Lpadding
        padding.append( (Lpadding,Rpadding) )
    return np.pad(arr,padding,constant_values=constant_values)


"""
##############################################################################
Marker detection
##############################################################################
"""

# given a depth image and an Nx2 array of pixel coordinates, returns a length N array
# of the depths at each of those pixels
def extract_depth_points(pixels,depth_img,depth_scale):
    depth_values = np.zeros(pixels.shape[0])
    for i in range(pixels.shape[0]): 
        p = pixels[i]
        depth_values[i] = depth_img[p[0],p[1]]

    return depth_values * depth_scale

# given the corners of a marker and a depth image, return a list of every pixel
# within the marker and the depth at those pixels
def extract_depth(corners,depth_img,depth_scale,show_plot=False):
    # generate an image where the points within the marker are colored
    poly = np.zeros(depth_img.shape)
    poly = cv2.fillConvexPoly(poly,cv2.convexHull(corners).astype('int32'),1)
    # get a list of [x,y] pixel coordinates for every pixel within the marker
    depth_pixels = np.column_stack(np.where(np.logical_and(poly>0,depth_img > 0)))
    
    real_depth_values = extract_depth_points(depth_pixels,depth_img,depth_scale)
    
    if show_plot:
        plt.figure()
        plt.imshow(color_img_pixels(depth_img,depth_pixels))
        plt.title("Depth image of marker")
        plt.show()
    
    return depth_pixels, real_depth_values

def color_img_pixels(img,pixels):
    disp_img = np.copy(img)
    for i in range(pixels.shape[0]): 
        p = pixels[i]
        disp_img[p[0],p[1]] = 255
    return disp_img

# least squares fit for depth, coeff would be in meters/pixel?
# pixels is Nx2, and values is length N with the idea that you would pass in
# the depth at each pixel for the values
def fit_depth(pixels,values):
    n_data = pixels.shape[0]
    # add an extra column of ones to allowing fitting with a z-intercept
    x = np.hstack((pixels,np.ones([n_data,1])))
    coeff,res,_,_ = np.linalg.lstsq(x,values,rcond=None)
    return coeff,res

# gets z value for a point on the fit plane with coefficients from fit_depth()
# works for arrays of the same shape as well
def point_from_plane(x,y,coeff):
    return coeff[0]*x + coeff[1]*y + coeff[2]

# given a set of 3D data, plots both the data and the plane fit of the data with
# the fit coefficients specified by coeff
# x_data, y_data, and z_data should fit what is expected by ax.scatter3D()
def plot_plane_fit(x_data,y_data,z_data,coeff):
    plt.figure()
    ax = plt.axes(projection='3d')
    x = np.linspace(min(x_data),max(x_data),10)
    y = np.linspace(min(y_data),max(y_data),10)
    X, Y = np.meshgrid(x, y)
    Z = point_from_plane(X,Y,coeff)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='k')
    ax.plot_wireframe(X, Y, Z,color='black')
    ax.scatter3D(x_data, y_data, z_data)
    plt.title("planar fit of marker depth data")
    plt.show()

# just a wrapper for rs.rs2_deproject_pixel_to_point()
# makes it easier to specify the way pixels are converted to camera coordinates
def pixel2camera(p1,p2,z):
    '''
    # an initial attempt used transformation matrices with homogeneous coordinates
    # this approach doesn't work
    inv_camera = np.copy(matrix_coefficients)
    inv_camera *= np.array([[1,0,-1],
                            [0,1,-1],
                            [0,0, 1]]) # reverse 2D translation
    inv_camera[[0,1],[0,1]] = z/(inv_camera[[0,1],[0,1]]) # reverse scaling to pixels
    return inv_camera @ np.array([[p1],[p2],[1]]) # 3x3 matrix times 3x1 column
    '''
    outp = rs.rs2_deproject_pixel_to_point(color_intr,[p2,p1],z)
    return [outp[1],
            outp[0],
            outp[2]]

# given the pixel locations of the corners of the charuco marker and the depth image,
# returns a 5x3 array of coordinates in the camera coordinate system. The first
# 4 rows are the coordinates of the corners, and the 5th row is the coordinate
# of the center of the marker.
# if plane_fit is True, the z-axis component of the marker center will be interpolated
# using a planar fit on the depth within the marker
def extract_charuco_depth(charucoCorners,img,depth_img,plane_fit = False,plotting = False):
    if plane_fit or plotting:
        pix,pix_depth = extract_depth(charucoCorners,depth_img,depth_scale)
        cam_xy = np.zeros(pix.shape)
        for i in range(pix.shape[0]):
            cam_xy[i,0],cam_xy[i,1],_ = pixel2camera(pix[i,0],pix[i,1],pix_depth[i])
        coeff,res = fit_depth(cam_xy,pix_depth)
    
    # why are these flipped but not the ones passed to fit_depth?
    # this rounding to an int32 reduces accuracy
    corner_pix = np.fliplr(charucoCorners.squeeze()).astype('int32')
    corner_d = extract_depth_points(corner_pix,depth_img,depth_scale)
    center_pix = np.mean(np.fliplr(charucoCorners.squeeze()),axis=0).astype('int32')
    
    
    corner_coord = np.zeros(corner_pix.shape)
    for i in range(corner_pix.shape[0]):
        corner_coord[i,0],corner_coord[i,1],_ = pixel2camera(corner_pix[i,0],corner_pix[i,1],corner_d[i])
    corner_coord = np.hstack((corner_coord,corner_d[:,np.newaxis]))
    
    if plane_fit:
        center_coord = np.mean(corner_coord[:,:2],axis=0)
        center_d = point_from_plane(center_coord[0],center_coord[1],coeff)
    else:
        center_d = extract_depth_points(center_pix[None,:],depth_img,depth_scale)
        center_coord = np.zeros(2)
        center_coord[0],center_coord[1],_ = pixel2camera(center_pix[0],center_pix[1],center_d)
    center_coord = np.hstack((center_coord,center_d))
    
    corner_coord = np.vstack((corner_coord,center_coord))

    #print("Corners:\n",corner_coord[:-1])
    print("Center:\n",corner_coord[-1])
    
    if plotting:
        plt.figure(figsize=[5,5])
        ll = np.min(corner_pix,axis=0) - 50
        ur = np.max(corner_pix,axis=0) + 50
        plt.imshow(img[ll[0]:ur[0],ll[1]:ur[1]])
        plt.title("detected marker")
        plt.show()
        plt.figure(figsize=[5,5])
        plt.imshow(depth_img[ll[0]:ur[0],ll[1]:ur[1]])
        plt.title("depth data of marker")
        plt.show()
        plot_plane_fit(cam_xy[:,0],cam_xy[:,1],pix_depth,coeff)
        plt.figure()
        plt.title("Side views of marker depth and marker center")
        plt.subplot(2,1,1)
        plt.plot(cam_xy[:,0],pix_depth,'ko',markersize=5)
        plt.plot(center_coord[0],center_d,'rx',markersize=10)
        plt.xlabel("x [mm]")
        plt.ylabel("depth [mm]")
        plt.subplot(2,1,2)
        plt.plot(cam_xy[:,1],pix_depth,'ko',markersize=5)
        plt.plot(center_coord[1],center_d,'rx',markersize=10)
        plt.xlabel("y [mm]")
        plt.ylabel("depth [mm]")
        plt.show()
    
    return corner_coord
    
# Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters_create()

# detects any aruco or charuco markers in an image
# returns the same image with any detected markers outlined, and an array of 
# corners if a charuco marker was detected
def detect_markers(img,depth_img=None, plotting = False):
    img = np.copy(img)
    
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        img, dictionary, parameters=parameters)
    
    if type(markerIds) == type(None): # no markers detected
        return img, np.array([])
    
    try:
        nCorners,charucoCorners,charucoIds = \
            aruco.interpolateCornersCharuco(markerCorners,markerIds,img,chboard)
        
        markerIds = np.squeeze(markerIds)
        
        aruco.drawDetectedMarkers(img, markerCorners)
        aruco.drawDetectedCornersCharuco(img,charucoCorners,charucoIds)
        
        # Estimate pose of each marker and use poses to draw marker axes
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(
            markerCorners, marker_side_length, matrix_coefficients,distortion_coefficients)
        for rv,tv in zip(rvec,tvec):
            aruco.drawAxis(img, matrix_coefficients, distortion_coefficients, 
                           rv, tv, marker_side_length/2)

        
        if nCorners == 4 and np.all(depth_img != None):
            corner_coord = extract_charuco_depth(charucoCorners,img,depth_img,plotting = plotting)
            return img, corner_coord
        
        if plotting:
            plt.figure(figsize=[20,10])
            plt.imshow(img)
            plt.show()
        return img, np.array([])
    
    except Exception as e:
        print("Marker detection failed: ",e)
        # comment this raise out if you don't want failed marker detection to
        # interrupt execution
        raise(e)
        return img, np.array([])


"""
##############################################################################
Marker calibration
##############################################################################
"""

# given two 4x3 arrays of pts, finds homogeneous transformation matrices between
# the two coordinate systems described by the points. pts should be ordered as 
# [center, center+x, center+y, center+z], where center is the origin, x is a point
# on the x-axis, y is a point on the y-axis, z is a point on the z-axis.
# transf_r2c converts from real to camera coordinates, transf_c2r is the inverse.
def get_camera_transform(camera_pts,true_pts):
    print("calibration points:")
    print(*zip(['origin:','x axis:','y axis:','z axis'],true_pts),sep='\n')
    print("Xcam Ycam Zcam vector norms:")
    xvec = (camera_pts[1] - camera_pts[0])
    xnorm = true_pts[1][0] - true_pts[0][0]
    print(np.linalg.norm(xvec), xnorm); xvec /= xnorm #np.linalg.norm(xvec)
    yvec = (camera_pts[2] - camera_pts[0])
    ynorm = true_pts[2][1] - true_pts[0][1]
    print(np.linalg.norm(yvec), ynorm); yvec /= ynorm #np.linalg.norm(yvec)
    zvec = (camera_pts[3] - camera_pts[0])
    znorm = true_pts[3][2] - true_pts[0][2]
    #zvec = np.cross(xvec,yvec)
    #znorm = np.linalg.norm(zvec)
    print(np.linalg.norm(zvec), znorm); zvec /= znorm #np.linalg.norm(zvec)
    print("Xcam Ycam Zcam vectors:")
    print(xvec)
    print(yvec)
    print(zvec)
    
    print("Angle between Xcam and Ycam:")
    print(np.arccos(np.dot(xvec/np.linalg.norm(xvec),yvec/np.linalg.norm(yvec)))*180/np.pi)
    print("Angle between Xcam and Zcam:")
    print(np.arccos(np.dot(xvec/np.linalg.norm(xvec),zvec/np.linalg.norm(zvec)))*180/np.pi)
    print("Angle between Ycam and Zcam:")
    print(np.arccos(np.dot(yvec/np.linalg.norm(yvec),zvec/np.linalg.norm(zvec)))*180/np.pi)
    
    rot_r2c = np.hstack((xvec[:,None],yvec[:,None],zvec[:,None]))
    #print(rot_r2c)
    transl_r2c = camera_pts[1][:,None] - (rot_r2c @ true_pts[1][:,None])
    transf_r2c = np.eye(4)
    transf_r2c[:3,:3] = rot_r2c
    transf_r2c[:3,3] = transl_r2c.squeeze()
    print("Real to camera transformation:")
    print(transf_r2c)

    rot_c2r = np.linalg.inv(rot_r2c)
    transl_c2r = - rot_c2r @ transl_r2c
    
    transf_c2r = np.eye(4)
    transf_c2r[:3,:3] = rot_c2r
    transf_c2r[:3,3] = transl_c2r.squeeze()
    
    print("Camera to real transformation:")
    print(transf_c2r)
    
    return transf_r2c,transf_c2r

def test_transform(transf_r2c,transf_c2r,pt,true_pt):
    print("Test real2cam")
    print(pt)
    pt_pred = apply_transf(transf_r2c,true_pt)
    print(pt_pred)
    print("Error:",np.linalg.norm(pt_pred - pt),'\n')
    
    print("Test cam2real")
    print(true_pt)
    true_pt_pred = apply_transf(transf_c2r,pt)
    print(true_pt_pred)
    print("Error:",np.linalg.norm(true_pt_pred - true_pt),'\n')

# this plots transformation errors for the homogeneous linear transformation 
# found using get_camera_transform. 
def plot_transform_errors(transf_c2r,pts,true_pts,calib_true_pts):
    fig = plt.figure(figsize = [6,6])
    ax = fig.gca(projection='3d')
    
    x,y,z = zip(*true_pts)
    
    true_pts_pred = np.copy(true_pts)
    for i in range(true_pts_pred.shape[0]):
        true_pts_pred[i] = apply_transf(transf_c2r,pts[i])
    
    u,v,w = zip(*(true_pts_pred - true_pts))
    
    ax.quiver(x, y, z, u, v, w, length=6)
    '''actual points'''
    ax.scatter3D(*zip(*true_pts), alpha=0.8, 
                 c=np.linalg.norm(true_pts_pred - true_pts,axis=-1))
    '''predicted points'''
    ax.scatter3D(*zip(*true_pts_pred), alpha=0.5, marker='D',
                 c=np.linalg.norm(true_pts_pred - true_pts,axis=-1))

    '''calibration points'''
    ax.scatter3D(*zip(*calib_true_pts),color='r',marker='X',s=60)
    plt.title("Linear calibration result")
    plt.show()

# converts coordinate to homogeneous coordinate:
# [1,2,3] -> [1,2,3,1]
def coord2homocoord(coord):
    return np.array([[coord[0]],[coord[1]],[coord[2]],[1]])

# applies a 4x4 homogeneous transformation to a 3D point. takes a 3 element array
# and returns a 3 element array.
def apply_transf(tf_matrix,coord):
    coord = np.squeeze(coord)
    return ( tf_matrix @ coord2homocoord(coord) )[:3].squeeze()

# creates a 4x4 homogeneous transformation matrix from a 3x3 rotation and a 
# 3x1 translation vector
def make_transf(rot_matrix,transl_vector):
    transf = np.eye(4)
    transf[:3,:3] = rot_matrix
    transf[:3,3] = transl_vector.squeeze()
    return transf

# find the transformed pts for a list of pts using apply_transf in a for-loop
def generate_prediction(transf_c2r,pts):
    pred = np.copy(pts)
    for i in range(pred.shape[0]):
        pred[i] = apply_transf(transf_c2r,pts[i])
    return pred


# we want to correct for the errors in the linear transformation found with
# get_camera_transform, so this function calculates the error and performs a
# non-linear fit. The nature of the non-linearity is determined by generate_fit_data.
def error_fit_calibration(transf_c2r,pts,true_pts):
    true_pts_pred = generate_prediction(transf_c2r,pts)
    
    coeff,_ = fit_calibration(true_pts_pred, true_pts)
    
    print("Error calibration coefficients:")
    print(coeff)
    
    fitted_pred = generate_fit_data(true_pts_pred) @ coeff
    
    print("Sum of squared errors on x, y, z:")
    print(np.sum((fitted_pred - true_pts)**2,axis=0))
    
    print("Max of squared errors on x, y, z:")
    print(np.max((fitted_pred - true_pts)**2,axis=0))
    
    print("Non-fitted max of squared errors on x, y, z:")
    print(np.max((true_pts_pred - true_pts)**2,axis=0))
    
    return coeff, fitted_pred

# generates plots of the calibration error that still exists even after applying
# the non-linear fit from error_fit_calibration. 
def plot_error_calibration_errors(transf_c2r,coeff,pts,true_pts):
    true_pts_pred = generate_prediction(transf_c2r,pts)
    fitted_pred = generate_fit_data(true_pts_pred) @ coeff
    
    fig = plt.figure(figsize = [6,6])
    ax = fig.gca(projection='3d')
    
    x,y,z = zip(*true_pts)
    
    u,v,w = zip(*(fitted_pred - true_pts))
    
    ax.quiver(x, y, z, u, v, w, length=5)
    '''actual points'''
    ax.scatter3D(*zip(*true_pts), alpha=0.8, 
                 c=np.linalg.norm(fitted_pred - true_pts,axis=-1))
    '''predicted points'''
    ax.scatter3D(*zip(*fitted_pred), alpha=0.5, marker='D',
                 c=np.linalg.norm(fitted_pred - true_pts,axis=-1))
    plt.title("Non-linear calibration result")
    plt.show()
    
    X_MIN,Y_MIN,Z_MIN = np.min(true_pts,axis = 0).astype('int32') - 70
    X_MAX,Y_MAX,Z_MAX = np.max(true_pts,axis = 0).astype('int32') + 70
    X_STEP,Y_STEP,Z_STEP = \
        ( (np.array([X_MAX,Y_MAX,Z_MAX]) - np.array([X_MIN,Y_MIN,Z_MIN]))/10 ).astype('int32')
    X,Y,Z = np.meshgrid(range(X_MIN,X_MAX,X_STEP),
                        range(Y_MIN,Y_MAX,Y_STEP),
                        range(Z_MIN,Z_MAX,Z_STEP))
    grid_pts = np.array([[x,y,z] for x,y,z in zip(X.flatten(),Y.flatten(),Z.flatten())])
    
    x = generate_fit_data(grid_pts)
    fitted_grid_pts = x @ coeff
    
    fig = plt.figure(figsize = [6,6])
    ax = fig.gca(projection='3d')
    x,y,z = zip(*grid_pts)
    u,v,w = zip(*(fitted_grid_pts - grid_pts))
    ax.quiver(x, y, z, u, v, w, length=5)
    plt.title("Error correction function")
    plt.show()
    
    scale = 1 # smaller = longer arrows
    fig = plot_quiver_slices(true_pts,true_pts_pred,angles='xy',scale_units='xy',scale=scale)
    axes = plt.gcf().get_axes()
    axes[0].scatter(fitted_pred[:,0],fitted_pred[:,1],c='y',marker='x')
    axes[1].scatter(fitted_pred[:,0],fitted_pred[:,2],c='y',marker='x')
    axes[2].scatter(fitted_pred[:,1],fitted_pred[:,2],c='y',marker='x')
    plot_quiver_slices(true_pts,fitted_pred-true_pts_pred+true_pts,fig=fig,angles='xy',scale_units='xy',scale=scale,color='red',alpha=0.7)
    #plot_quiver_slices(grid_pts,fitted_grid_pts,fig=fig,angles='xy',scale_units='xy',scale=scale,color='blue',alpha=0.7)
    
    for ax in axes:
        ax.margins(0.15)
    plt.show()

# plots 2D views of a 3D vector field. the pts arrays are Nx3, and the vectors
# will point from pts in from_pts to to_pts.
# if fig is given, these views will be plotted on the figure, else a new figure
# will be created. 
def plot_quiver_slices(from_pts,to_pts,fig = None, **kwargs):
    x,y,z = zip(*from_pts)
    u,v,w = zip(*(to_pts - from_pts))
    
    if fig == None:
        fig = plt.figure(figsize = [18,6])
        plt.subplot(1,3,1)
        plt.title("Calibration error in the xy-plane")
        plt.quiver(x, y, u, v,**kwargs)
        plt.xlabel('x'); plt.ylabel('y')
        plt.gca().set_xticks(np.unique(x))
        plt.gca().set_yticks(np.unique(y))
        plt.grid()
        plt.subplot(1,3,2)
        plt.title("Calibration error in the xz-plane")
        plt.quiver(x, z, u, w,**kwargs)
        plt.xlabel('x'); plt.ylabel('z')
        plt.gca().set_xticks(np.unique(x))
        plt.gca().set_yticks(np.unique(z))
        plt.grid()
        plt.subplot(1,3,3)
        plt.title("Calibration error in the yz-plane")
        plt.quiver(y, z, v, w,**kwargs)
        plt.xlabel('y'); plt.ylabel('z')
        plt.gca().set_xticks(np.unique(y))
        plt.gca().set_yticks(np.unique(z))
        plt.grid()
    else:
        axes = plt.gcf().get_axes()
        axes[0].quiver(x, y, u, v,**kwargs)
        axes[1].quiver(x, z, u, w,**kwargs)
        axes[2].quiver(y, z, v, w,**kwargs)
    
    return fig
    
def print_worst_fit_points(fitted_pts,true_pts):
    print("\nThe 10 points with the highest calibration errors:")
    fitted_indx = sorted(range(fitted_pts.shape[0]),
                         key=lambda i:np.linalg.norm(fitted_pts[i] - true_pts[i]),
                         reverse=True)
    
    for i in fitted_indx[:10]:
        print("Point %d error:"%i)
        print(fitted_pts[i] - true_pts[i])
        print(np.linalg.norm(fitted_pts[i] - true_pts[i]))
    
    fitted_err = np.linalg.norm(fitted_pts - true_pts,axis=1)
    
    plt.figure()
    plt.hist(np.linalg.norm(fitted_pts - true_pts,axis=1),bins='fd')
    plt.title("Calibration error per point")
    plt.xlabel("Error [mm]")
    plt.ylabel("Count")
    plt.show()
    
    print("\n")
    return fitted_indx, fitted_err

# Creates the data that is used for non-linear error correction.
def generate_fit_data(pts):
    # make a row vector if not already one
    if pts.ndim < 2:
        pts = pts[None,:]
    # add an extra column of ones to allowing fitting with a z-intercept, and
    # add columns of squared values for 2nd order fit
    return np.hstack((pts**2,pts,np.ones([pts.shape[0],1])))
    # add cross terms for good measure, in the order xy, xz, yz
    #return np.hstack((pts[:,(0,0,1)]*pts[:,(1,2,2)],pts**2,pts,np.ones([pts.shape[0],1])))

# performs a linear least-squares fit. for a non-linear fit, the augmented data
# from generate_fit_data is used for pts_from.
def fit_calibration(pts_from,pts_to):
    x = generate_fit_data(pts_from)
    coeff,res,_,_ = np.linalg.lstsq(x,pts_to,rcond=None)
    return coeff,res

# converts a pixel location on the realsense camera to a coordinate that can be
# used to position the gantry.
def pixel_to_gantry(x,y,depth_img):
    # get depths from depth image
    depths = extract_depth_points(np.array([[x,y]],dtype='int32'),depth_img,1)
    # convert to camera coordinate system
    cam_coord = np.zeros(3)
    cam_coord[0],cam_coord[1],_ = pixel2camera(x,y,depths[0])
    cam_coord[2] = depths[0]
    # apply linear homogeneous transformation
    gantry_coord = apply_transf(tf_c2r,cam_coord)
    # correct for errors in homogeneous transformation with non-linear fit
    x_fit = generate_fit_data(gantry_coord)
    corrected_coord = x_fit @ error_coeff
    
    return corrected_coord

def test_pixel_to_gantry(i,true_pts = None):
    img, dimg = load_images(i,folder = CALIB_FOLDER)
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        img, dictionary, parameters=parameters)
    nCorners,charucoCorners,charucoIds = \
            aruco.interpolateCornersCharuco(markerCorners,markerIds,img,chboard)
    center_pix = np.mean(np.fliplr(charucoCorners.squeeze()),axis=0).astype('int32')
    
    gantry_coord = pixel_to_gantry(center_pix[0],center_pix[1],dimg)
    
    print("Test pixel to gantry on:\n%s"%center_pix)
    print("Result: %s"%gantry_coord)
    if true_pts != None:
        print("Expected: %s"%true_pts[i])
        print("Error: %f"%np.linalg.norm(gantry_coord - true_pts[i]))


def load_npz(fname,folder = "."):
    npz = np.load(os.path.join(folder,fname))
    return npz[npz.keys().__iter__().send(None)]

def load_images(n,folder = "."):
    return cv2.imread(os.path.join(folder,"color%d.png"%n)), \
        load_npz("depth%d.npz"%n,folder=folder)

def write_pts(arr,fname,folder='.'):
    np.savetxt(os.path.join(folder,fname), arr, fmt='%1.8f')
    
def print_new_calibration_matrices(tf_c2r,coeff):
    print("Paste the following matrix into tf_c2r:")
    print(tf_c2r.tolist())
    print("Paste the following matrix into error_coeff:")
    print(coeff.tolist())


if __name__ == "__main__":
    true_pts = np.loadtxt(os.path.join(CALIB_FOLDER,"gantry_coord.txt"))
    calib_true_pts = true_pts[CALIB_POINT_INDEX]
    
    cam_pts = []
    try:
        if RECALCULATE:
            if len(true_pts) > 100:
                confirm = "y" == input(
                "Are you sure you want to recalculate for %d points? (y/n): "%len(true_pts)
                )
            else:
                confirm = True
            
            if confirm:
                raise(Exception("Not attempting to load from file"))
                
        cam_pts = np.loadtxt(os.path.join(CALIB_FOLDER,"camera_pts.txt"))
    except:
        for i in range(true_pts.shape[0]):
            img, dimg = load_images(i,folder = CALIB_FOLDER)
            
            if UNDISTORT:
                img = opencv_camera_calibration.undistort_color(img)
                dimg = opencv_camera_calibration.undistort_color(dimg)

            _, coord = detect_markers(img,dimg,plotting=False)
            if len(coord) != 0:
                cam_pts.append(coord[-1])
            else:
                cam_pts.append([np.nan,np.nan,np.nan])
        write_pts(cam_pts,"camera_pts.txt",folder=CALIB_FOLDER)
    
    calib_pts = [ cam_pts[i] for i in CALIB_POINT_INDEX ]
    
    for i in range(len(cam_pts)):
        if cam_pts[i][0] is np.nan:
            SKIP_POINTS.append(i)
            print("Marker not detected in image %d"%i)
    cam_pts = np.delete(cam_pts,SKIP_POINTS,axis=0)
    true_pts = np.delete(true_pts,SKIP_POINTS,axis=0)
    
    
    
    tf_r2c, tf_c2r = get_camera_transform(calib_pts,calib_true_pts)
    
    #test_transform(tf_r2c, tf_c2r, calib_pts[1],calib_true_pts[1])
    
    test_transform(tf_r2c, tf_c2r, cam_pts[-1],true_pts[-1])

    plot_transform_errors(tf_c2r,cam_pts,true_pts,calib_true_pts)
    
    plot_transform_errors(tf_r2c,true_pts,cam_pts,calib_pts)

    coeff, fitted_pred = error_fit_calibration(tf_c2r,cam_pts,true_pts)
    
    plot_error_calibration_errors(tf_c2r,coeff,cam_pts,true_pts)
    
    fitted_indx,fitted_err = print_worst_fit_points(fitted_pred,true_pts)
    
    print_new_calibration_matrices(tf_c2r,coeff)