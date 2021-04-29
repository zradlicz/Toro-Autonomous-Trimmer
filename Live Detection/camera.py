# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:02:37 2021

@author: zradlicz
"""
import pyrealsense2 as rs
import numpy as np
import cv2

USE_CAMERA = True
bag_file = "C:/Users/zradlicz/Desktop/Jupyter/ME 4054/Live_detection/latest.bag"
DEPTH_RESOLUTION = (848,480)
COLOR_RESOLUTION = (1280,720)

def cresolution():
    return COLOR_RESOLUTION

def dresolution():
    return DEPTH_RESOLUTION

def create_pipeline():
    # Create a pipeline
    pipeline = rs.pipeline()
    
    
    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    if not USE_CAMERA:
        rs.config.enable_device_from_file(config, bag_file)
    config.enable_stream(rs.stream.depth, DEPTH_RESOLUTION[0], DEPTH_RESOLUTION[1], rs.format.z16, 30) # max is 1280x720
    #config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
    #config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 15)
    config.enable_stream(rs.stream.color, COLOR_RESOLUTION[0], COLOR_RESOLUTION[1], rs.format.rgb8, 30) # max is 1920x1080
    
    # Start streaming
    profile = pipeline.start(config)
    
    color_sensor = profile.get_device().query_sensors()[1]
    #infrared_sensor = profile.get_device().query_sensors()[0]
    #color_sensor.set_option(rs.option.exposure, 156)
    #color_sensor.set_option(rs.option.white_balance, 4600)
    #color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    #color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
    #color_sensor.set_option(rs.option.gain, 64)
    
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    return pipeline,align,depth_scale,color_sensor


def get_rs_filters():
    # filters reference: https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 4)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5) # .25 to 1, higher is less filter
    spatial.set_option(rs.option.filter_smooth_delta, 20) # 1 to 50
    #spatial.set_option(rs.option.holes_fill, 3)
    
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.5) # .25 to 1, higher is less filter
    temporal.set_option(rs.option.filter_smooth_delta, 10) # 1 to 50
    
    hole_filling = rs.hole_filling_filter()
    
    return [spatial,temporal,hole_filling]


def get_frames(pipeline,align,filters):
    frames = pipeline.wait_for_frames()
    # Align the depth frame to color frame NOTE: this is SPATIAL alignment, so no fussing with coordinate systems needed
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    
    # apply depth filters
    for filte in filters:
        aligned_depth_frame = filte.process(aligned_depth_frame)


    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    #depth_image = np.copy(depth_image_raw)
    #depth_image = depth_image[:,184:664]
    #depth_image = cv2.resize(depth_image,(256,256))

    
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    #color_image = np.copy(color_image_raw)
    #color_image = color_image[:,280:1000,:]
    #color_image = cv2.resize(color_image,(256,256))
    return color_image,depth_image,color_frame,aligned_depth_frame