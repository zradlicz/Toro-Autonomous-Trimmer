# -*- coding: utf-8 -*-
"""
Created on Sun May  2 12:42:52 2021

@author: Alan

Contains all code needed to interface with the Intel Realsense camera. This
include instantiating objects, getting frames, aligning and filtering frames,
recording frames to .bag files, and changing some of the camera settings.

The __main__ block at the end of this file contains some example code. Run this 
file to see it in action.

keypresses recognized during execution are as follows:
q : quit
z : start recording if RECORD = True
x : pause recording if RECORD = True
v : save a snapshot of the currently displayed frame
yuiokl : adjusts color camera settings
"""

#Library for Intel RealSense, documentation can be found at https://dev.intelrealsense.com/docs/python2
import pyrealsense2 as rs
import numpy as np
import cv2

color_sensor_default = {rs.option.exposure: 156,
            rs.option.white_balance: 4600,
            rs.option.enable_auto_exposure: 1,
            rs.option.enable_auto_white_balance: 1,
            rs.option.gain: 64}

class RealsenseCamera:    
    def __init__(self,USE_CAMERA = True, RECORD = False, bag_file = "./bags/default.bag",
                 record_file = "./bags/output.bag", DEPTH_RESOLUTION = (848,480),
                 COLOR_RESOLUTION = (1280,720), FRAME_RATE = 15,
                 color_sensor_settings = color_sensor_default):
        self.USE_CAMERA = USE_CAMERA
        self.RECORD = RECORD
        self.bag_file = bag_file
        self.record_file = record_file
        self.DEPTH_RESOLUTION = DEPTH_RESOLUTION
        self.COLOR_RESOLUTION = COLOR_RESOLUTION
        self.FRAME_RATE = FRAME_RATE
        
        self.create_pipeline()
        self.set_sensor_settings(color_sensor_settings)
        self.get_rs_filters()
        self.colorizer = rs.colorizer()
        
        if self.RECORD:
            self.get_rs_recorder()
            print("Recording...")
        
        self.print_pipeline_info()
        
        self.expo = self.color_sensor.get_option(rs.option.exposure)
        self.wb = self.color_sensor.get_option(rs.option.white_balance)
        self.gain = np.log2(self.color_sensor.get_option(rs.option.gain))
        
        self.num_snapshots = 0
        
        # Skip 5 first frames to give the Auto-Exposure time to adjust
        for i in range(5):
            self.pipeline.wait_for_frames()
    
    def create_pipeline(self):
        # Create a pipeline
        pipeline = rs.pipeline()
        
        #Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()
        if not self.USE_CAMERA:
            rs.config.enable_device_from_file(config, self.bag_file)
        if self.RECORD:
            config.enable_record_to_file(self.record_file)
        
        if self.USE_CAMERA:
            config.enable_stream(rs.stream.depth, 
                                 self.DEPTH_RESOLUTION[0], self.DEPTH_RESOLUTION[1], 
                                 rs.format.z16, self.FRAME_RATE) # supposedly max is 1920x1080, but its actually 848x480
            config.enable_stream(rs.stream.color, 
                                 self.COLOR_RESOLUTION[0], self.COLOR_RESOLUTION[1], 
                                 rs.format.rgb8, self.FRAME_RATE) # max is 1280x720
            config.enable_stream(rs.stream.infrared, 848, 480, rs.format.any, self.FRAME_RATE)
        else:
            config.enable_all_streams()
        
        # Start streaming
        profile = pipeline.start(config)
        
        # get the color and infrared sensors
        color_sensor = profile.get_device().query_sensors()[1]
        infrared_sensor = profile.get_device().query_sensors()[0]
        
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)
        
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        self.pipeline = pipeline
        self.align = align
        self.depth_scale = depth_scale
        self.color_sensor = color_sensor
        self.infrared_sensor = infrared_sensor
    
    def set_sensor_settings(self,color_settings):
        if self.USE_CAMERA:
            for setting, value in color_settings:
                self.color_sensor.set_option(setting, value)
    
    def get_rs_filters(self):
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
        
        self.filters = [spatial,temporal,hole_filling]

    def get_rs_recorder(self):
        self.recorder = self.pipeline.get_active_profile().get_device().as_recorder()
    
    def print_pipeline_info(self):
        # print depth frame resolution
        frames = self.pipeline.wait_for_frames()
        try:
            print("Depth resolution: ", np.asanyarray(frames.get_depth_frame().get_data()).shape)
            print("RGB resolution: ", np.asanyarray(frames.get_color_frame().get_data()).shape)
            print("IR resolution: ", np.asanyarray(frames.get_infrared_frame().get_data()).shape)
        except:
            pass
        print("Frames per frames: ", frames.size())
        
        for stream in self.pipeline.get_active_profile().get_streams():
            streamvs = stream.as_video_stream_profile()
            print(streamvs)
            print(streamvs.get_intrinsics())
    
    def cresolution(self):
        return self.COLOR_RESOLUTION

    def dresolution(self):
        return self.DEPTH_RESOLUTION
    
    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        # Align the depth frame to color frame
        # NOTE: this is SPATIAL alignment, so no fussing with extrinsic coordinate systems needed
        aligned_frames = self.align.process(frames)
    
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        #ir_frame = aligned_frames.get_infrared_frame()
        
        # apply depth filters
        raw_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        interm = []
        for filte in self.filters:
            interm.append(np.asanyarray(self.colorizer.colorize(aligned_depth_frame).get_data()))
            aligned_depth_frame = filte.process(aligned_depth_frame)
    
    
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        #depth_image = cv2.resize(depth_image,(256,256))
        depth_colormap = np.asanyarray(self.colorizer.colorize(aligned_depth_frame).get_data())
    
        
        color_image = np.asanyarray(color_frame.get_data())
        #color_image = cv2.resize(color_image,(256,256))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        #ir_image = np.asanyarray(ir_frame.get_data())
        
        self.color_image = color_image
        self.interm = interm
        self.raw_depth_image = raw_depth_image
        self.depth_image = depth_image
        self.depth_colormap = depth_colormap
        #self.ir_image = ir_image
    
    def save_snapshot(self,i):
        cv2.imwrite("color%d.png"%i,self.color_image)
        np.savez_compressed("depth%d"%i,self.depth_image)
        np.savez_compressed("depth_raw%d"%i,self.raw_depth_image)
        print("Wrote images number %d"%i)
    
    def process_keypress(self,key,ignore = ''):
        if chr(key & 0xFF) in ignore:
            return
        if self.RECORD:
            if key & 0xFF == ord('z'):
                print("Recording...")
                self.recorder.resume()
            
            if key & 0xFF == ord('x'):
                print("Recording paused")
                self.recorder.pause()
        
        if key & 0xFF == ord('v'):
            self.save_snapshot(self.num_snapshots)
            self.num_snapshots += 1
        
        try:
            if key & 0xFF == ord('y'):
                self.expo += 30
                print("Exposure:",self.expo)
                self.color_sensor.set_option(rs.option.exposure, self.expo)
            if key & 0xFF == ord('u'):
                self.expo -= 30
                print("Exposure:",self.expo)
                self.color_sensor.set_option(rs.option.exposure, self.expo)
            if key & 0xFF == ord('i'):
                self.wb += 300
                print("White balance:",self.wb)
                self.color_sensor.set_option(rs.option.white_balance, self.wb)
            if key & 0xFF == ord('o'):
                self.wb -= 300
                print("White balance:",self.wb)
                self.color_sensor.set_option(rs.option.white_balance, self.wb)
            if key & 0xFF == ord('k'):
                self.gain += 1
                print("gain:",2**self.gain)
                self.color_sensor.set_option(rs.option.gain, 2**self.gain)
            if key & 0xFF == ord('l'):
                self.gain -= 1
                print("gain:",2**self.gain)
                self.color_sensor.set_option(rs.option.gain, 2**self.gain)
        except:
            print("ERROR: Invalid camera parameter setting")
    
    def stop(self):
        self.pipeline.stop()
        if self.RECORD:
            del self.recorder
        del self.pipeline


def render(**kwargs):
    # kwargs should be image_name:image_array
    for name in kwargs:
        cv2.imshow(name,kwargs[name])

if __name__ == "__main__":
    import aruco_calibration as arct
    cam = RealsenseCamera(USE_CAMERA = False, bag_file = "./bags/markers_angled.bag")
    
    try:
        while True:
            cam.get_frames()
            
            color_image_disp = np.copy(cam.color_image)
            render(aruco = arct.detect_markers(color_image_disp,depth_img=cam.depth_image)[0],
                   #arucoIR = arct.detect_markers(ir_image)[0],
                   depth = cam.depth_colormap)
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break
            
            cam.process_keypress(key)
    finally:
        cv2.destroyAllWindows()
        cam.stop()

