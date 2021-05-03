# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:09:24 2021

@author: Alan

Run this file for automated calibration data collection. The data saved can be 
directly used with aruco_calibration.py.

The basic idea:
1. a list of coordinates are generated
2. move trimmer to a coordinate
3. wait until a charuco marker is detected in the camera image
4. save color image, depth image, raw depth image, and position of the gantry
5. repeat 2 - 4 until all coordinates visited

keypresses recognized during execution are as follows:
q : quit
wasdrf : rough positioning of gantry - wasd for xy-plane and rf to move along z-axis
g : this script waits until all 4 corners of a marker is detected. 'g' skips this waiting
And all camera keypress with the exception of 'v'
"""

# calibration parameters
X_MIN = 25
X_MAX = 240
X_STEP = 50
Y_MIN = 25
Y_MAX = 450
Y_STEP = 100
Z_MIN = 12
Z_MAX = 110
Z_STEP = 25

# data saved is numbered starting from this index, useful for adding more data
# to a preexisting calibration dataset
INITIAL_NUMBER = 0

USE_ARDUINO = True
GEN_CALIB = True

import pyrealsense2 as rs
import numpy as np
import cv2

import camera

from Calibration import aruco_calibration as arcc

if USE_ARDUINO:
    import arduino as ta


# creates array of coordinates to visit to collect calibration data
def generate_coordinates():
    if not GEN_CALIB:
        return []
    X,Y,Z = np.meshgrid(range(X_MIN,X_MAX,X_STEP),
                        range(Y_MIN,Y_MAX,Y_STEP),
                        range(Z_MIN,Z_MAX,Z_STEP))
    outp = np.array([[x,y,z] for x,y,z in zip(X.flatten(),Y.flatten(),Z.flatten())])
    print("Number of calibration points:",len(outp))
    print("Estimated total calibration data size:",2.145*len(outp),"MB")
    if input("Go calibration? (y/n): ") == 'y':        
        return outp
    else:
        return []

if __name__ == "__main__":
    calib_color_settings = dict(camera.color_sensor_default)
    calib_color_settings[rs.option.enable_auto_exposure] = 0
    calib_color_settings[rs.option.enable_auto_white_balance] = 0
    calib_color_settings[rs.option.gain] = 64
    
    cam = camera.RealsenseCamera(USE_CAMERA = True,color_sensor_settings=calib_color_settings)
        
    trimmer_coord = [0,0,108]
    coords = []
    calib_coords = generate_coordinates()
    coord_counter = 0
    debounce = 0
    
    marker_wait_frames = 3
    marker_wait_counter = marker_wait_frames
    
    # Streaming loop
    try:        
        if USE_ARDUINO:
            trimmer = ta.TrimmerArduinoNoblock('COM7',115200,1)
            trimmer.start_connection()
            while not trimmer.waiting:
                trimmer.start_connection()
                trimmer.update()
        
        num_written = INITIAL_NUMBER
        while True:
            cam.get_frames()
            
            color_image_disp = np.copy(cam.color_image)
            camera.render(color = cam.color_image)
            aruco_image, corners = arcc.detect_markers(color_image_disp,depth_img=cam.depth_image)
            camera.render(aruco = aruco_image, depth = cam.depth_colormap)
            
            if USE_ARDUINO:
                trimmer.update()

            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                break
            
            cam.process_keypress(key,ignore = 'wasd'+'rf'+'g'+'v')
            
            if USE_ARDUINO and debounce == 0:
                if key & 0xFF == ord('d'):
                    debounce = 5
                    trimmer_coord[0] += 20
                    trimmer.write_gcode('G00',X=trimmer_coord[0],Y=trimmer_coord[1],Z=trimmer_coord[2])
                if key & 0xFF == ord('a'):
                    debounce = 5
                    trimmer_coord[0] -= 20
                    trimmer.write_gcode('G00',X=trimmer_coord[0],Y=trimmer_coord[1],Z=trimmer_coord[2])
                if key & 0xFF == ord('w'):
                    debounce = 5
                    trimmer_coord[1] += 20
                    trimmer.write_gcode('G00',X=trimmer_coord[0],Y=trimmer_coord[1],Z=trimmer_coord[2])
                if key & 0xFF == ord('s'):
                    debounce = 5
                    trimmer_coord[1] -= 20
                    trimmer.write_gcode('G00',X=trimmer_coord[0],Y=trimmer_coord[1],Z=trimmer_coord[2])
                if key & 0xFF == ord('r'):
                    debounce = 5
                    trimmer_coord[2] += 10
                    trimmer.write_gcode('G00',X=trimmer_coord[0],Y=trimmer_coord[1],Z=trimmer_coord[2])
                if key & 0xFF == ord('f'):
                    debounce = 5
                    trimmer_coord[2] -= 10
                    trimmer.write_gcode('G00',X=trimmer_coord[0],Y=trimmer_coord[1],Z=trimmer_coord[2])
            else:
                debounce -= 5
            
            # if key & 0xFF == ord('p'):
            #     coord = np.zeros(3)
            #     corner_coord = np.array([camt.cX,camt.cY])
                
            #     gantry_coord = arcc.pixel_to_gantry(corner_coord[0],corner_coord[1],depth_image)
            #     gantry_coord = gantry_coord.squeeze()
            #     gantry_coord[0] += 132.8
            #     gantry_coord[1] += 38.02
            #     gantry_coord[2] += -3.24
            #     gantry_coord = np.round(gantry_coord,decimals=3)
                
            #     print("Calculated coordinates:",'#'*40)
            #     print(gantry_coord)
                
            #     if USE_ARDUINO:
            #         trimmer.write_gcode('G00',Z=108)
            #         trimmer.write_gcode('G00',X=gantry_coord[0],
            #                             Y=gantry_coord[1],
            #                             Z=108)
            #         trimmer.write_gcode('G00',X=gantry_coord[0],
            #                             Y=gantry_coord[1],
            #                             Z=gantry_coord[2])
            
            skip_waiting = False
            if key & 0xFF == ord('g'):
                skip_waiting = True
            
            if USE_ARDUINO and coord_counter < len(calib_coords):
                if trimmer.trimmer_flag == 'report':
                    if np.all(corners.shape==(5,3)) or skip_waiting:
                        # throw away first few detected frames to allow image to settle
                        if marker_wait_counter > 0 and not skip_waiting:
                            marker_wait_counter -= 1
                            continue
                        else:
                            marker_wait_counter = marker_wait_frames
                        # write image data
                        cam.save_snapshot(num_written)
                        num_written += 1
                        # record gantry coordinate
                        coords.append(calib_coords[coord_counter])
                        coord_counter += 1
                        # don't write more gcode if all coordinates visited
                        if coord_counter == len(calib_coords):
                            continue
                        # gcode: goto coordinate, send report when done, reset flag
                        trimmer.write_gcode('G00',X=calib_coords[coord_counter][0],
                                            Y=calib_coords[coord_counter][1],
                                            Z=calib_coords[coord_counter][2])
                        trimmer.write_gcode('P1')
                        trimmer.trimmer_flag = 'going'
                elif trimmer.trimmer_flag == '' and coord_counter == 0:
                    # only runs once for the first coordinate visited
                    trimmer.write_gcode('G00',X=calib_coords[coord_counter][0],
                                            Y=calib_coords[coord_counter][1],
                                            Z=calib_coords[coord_counter][2])
                    trimmer.write_gcode('P1')
                    trimmer.trimmer_flag = 'going'
    finally:
        cv2.destroyAllWindows()
        cam.stop()
        
        if USE_ARDUINO:
            trimmer.close()
        
        if len(coords) != 0:
            np.savetxt("gantry_coord%d.txt"%INITIAL_NUMBER, np.array(coords), fmt='%d')