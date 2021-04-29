# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:18:44 2021

@author: zradlicz
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:13:58 2021
@author: zradlicz
"""

USE_TF = True #CHNAGE THIS VALUE TO FALSE IF YOU DONT WANT TO USE TENSORFLOW (NO PREDICTION)
USE_ARDUINO = False  #CHNAGE THIS VALUE TO TRUE IF CONNECTED TO ARDUINO
#DISPSIZE = 700
#IMG_SIZE = 256

import pyrealsense2 as rs  #Library for Intel RealSense, documentation can be found at https://dev.intelrealsense.com/docs/python2
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time


if USE_TF:
    import tensorflow as tf

if USE_ARDUINO:
    import Arduino as ta #ta is for TrimmerArduino

import camera as cm
import aruco as ar
import toolpath as tp
import probe as pb  #These are the files we made that AutoTrim needs to run



def get_prediction(model,img): #get prediction gets the prediction from the tensorflow model defined on line 164
    if USE_TF:
        ysize,xsize,_ = img.shape
        base = np.zeros((ysize,xsize))
        img = img[:,int((xsize-ysize)/2):int(xsize-(xsize-ysize)/2),:] #cutting off enough of each side of the image so that it is a square
        img = cv2.resize(img,(256,256)) #model is trained on 256x256 sixed images, so it must predict on that size as well
        img = img/255.0 #image must be normalized to between 0 and 1
        img = np.array(img).reshape(-1, 256, 256, 3) #this is the shape needed to make a prediction
        prediction = model.predict(img)
        pred = prediction[0]
        pred = (pred[:,:,0]+pred[:,:,1]+pred[:,:,2])/3
        pred = cv2.resize(pred,(ysize,ysize))
        base[:,int((xsize-ysize)/2):int(xsize-(xsize-ysize)/2)] = pred #placing the 256x256 prediction into an image the same size as the original image
    else:
        pred = plt.imread("./test.png")/255
        pred = cv2.resize(pred,(256,256))
    return base

def depth_prediction(img,val): #Function is deprecated, no longer needed to be used
    ysize,xsize = img.shape
    base = np.zeros((ysize,xsize))
    mod = np.copy(img)
    smallimg = mod[:,int((xsize-ysize)/2):int(xsize-(xsize-ysize)/2)]
    maxval = np.max(smallimg)
    smallimg[smallimg<maxval*val] = 0
    base[:,int((xsize-ysize)/2):int(xsize-(xsize-ysize)/2)] = smallimg
    return base
    
def threshold(img,val): #Threshold funtion to get a mask made of stricly 1s and 0s
    #the closer val is to 1, the less of the prediction will appear in the threshold prediction
    thresh = np.max(img)*val
    #thresh=.4
    thresh_pred = np.copy(img)
    thresh_pred[thresh_pred>=thresh] = 1
    thresh_pred[thresh_pred<thresh] = 0
    
    return thresh_pred


def render(**kwargs): #render the images in multiple windows
    # kwargs should be image_name:image_array, maybe kwargs isn't the best choice when I could just use a regular dictionary...
    for name in kwargs:
        #cv2.imshow(name,cv2.resize(kwargs[name],(DISPSIZE, DISPSIZE),interpolation=cv2.INTER_AREA))
        cv2.imshow(name,kwargs[name])
        cv2.setMouseCallback(name, click_and_go) #this function allows for the click_and_go function to work
        

def successful_trim(cnt,val): #makes bouding shape either green or red depending on how close it is to shape of segmentation
    #the smaller val is, the closer the shape has to be to a circle to be successful
    (xcirc,ycirc),radius = cv2.minEnclosingCircle(cnt)
    black_image = np.zeros((cm.cresolution()[1],cm.cresolution()[0]))    
    black_image = cv2.circle(black_image,(int(xcirc),int(ycirc)),int(radius),(255,0,0),-1)
    black_image = black_image.astype('uint8')

    cnts,hierarchy=cv2.findContours(black_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    circle_cnt = max(cnts, key=cv2.contourArea)
    acc = cv2.matchShapes(cnt,circle_cnt,1,0.0) #this is the shape comparison function from cv2
    if(acc<val):
        succ = True
    else:
        succ = False
    return succ


def click_and_go(event, x, y, flags, params): #function that allows you to click on a point in the image, and have the gantry go there
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        real_coords = pix2real(x,y,depth_image)
        #print(x,y)
        print(real_coords) #can print real coords (gantry space) or x,y (pixel space)
       
        Trimmer.write_np_to_arduino(real_coords) #writes the coordinates to the gantry
        Trimmer.update()


def pix2real(x,y,depth_image): #converts pixel space to gantry space
    #the comments in this are left in so you can see what aruco is doing
    val = ar.pixel_to_gantry(y,x,depth_image)
    
    #    matrix_coefficients = np.array([[916.05,0,637.569], # for RGB camera
#                                [0,916.009,363.627],
#                                [0,0,1]])
#
#    
#    tf_matrix = np.array([[-0.9565677950546827,
#                          -0.016546617346755377,
#                          -0.023484550026097402,
#                          109.2423020685986],
#                         [0.0033351132754657112,
#                          -0.9575209468227942,
#                          -0.004131467301901673,
#                          290.8339547952038],
#                         [-0.009016358671417138,
#                          -0.002414404289663443,
#                          -0.9436274780342566,
#                          680.0184037204939],
#                         [0.0, 0.0, 0.0, 1.0]]
#                        )
#    
#    coord = np.zeros(3)
#    corner_coord = np.array([y,x])
#    depths,_  = ar.extract_depth_points(corner_coord[:2][None,:].astype('int32'),depth_image,1)
#    coord[0],coord[1],_ = ar.pixel2real_camera(corner_coord[0],corner_coord[1],depths[0],matrix_coefficients)
#    coord[2] = depths[0]
#    val = apply_transf(tf_matrix,coord)
#    #val[0] += 98 #x adjust
#    #val[1] += -35 #y adjust
#    val[0] += 0
#    val[1] += 0
    
    ##########################################
    #THESE VALUES ARE CRITICAL
    #x adjust is the distance in mm along the x axis from the marker calibration point to the center of the trimmer head
    #y adjust is the same but in y direction
    #z adjust is the same but in z direction
    ##########################################
    val[0][0] += 100#x adjust
    val[0][1] += 17 #y adjust
    val[0][2] -= 10 #z adjust
    ###########################################
    return val
        

def apply_transf(tf_matrix,coord):    #function is deprecated, but left in just in case you want to not use aruco for transformation
    return ( tf_matrix @ np.array([[coord[0]],
                                    [coord[1]],
                                    [coord[2]],
                                    [1]]      ) )[:3].squeeze()
    
    


if __name__ == "__main__": #start of the main loop

    if USE_TF:
        model = tf.keras.models.load_model("segmentation_256_200_rgb.h5") #the model we have been using for detection, feel free to train your own
    else:
        model = None
    
    pipeline,align,depth_scale,color_sensor = cm.create_pipeline() #these three lines are where camera.py is used
    colorizer = rs.colorizer()
    filters = cm.get_rs_filters()
    time.sleep(3)
    
    # Streaming loop
    try:
        if USE_ARDUINO:
            Trimmer = ta.TrimmerArduinoNoblock('COM3',115200,1) #See arduino for info about this Trimmer object
            #Trimmer = ta.TrimmerArduino('COM3',115200)
            Trimmer.start_connection()
                
        # print depth frame resolution
        frames = pipeline.wait_for_frames()
        print(np.asanyarray(frames.get_depth_frame().get_data()).shape)
        
        for x in range(5): # Skip 5 first frames to give the Auto-Exposure time to adjust
            frames = pipeline.wait_for_frames()
            
        expo = color_sensor.get_option(rs.option.exposure) #these will be used for manula adjustments later
        wb = color_sensor.get_option(rs.option.white_balance)
        gain = np.log2(color_sensor.get_option(rs.option.gain))
        
        
        while True:
            
            #Main loop, follow along with flowchart, the match pretty closely
            color_image,depth_image,color_frame,depth_frame = cm.get_frames(pipeline,align,filters) #get different types of images from camera
            test = np.copy(color_image)
            prediction = get_prediction(model,color_image) #use the model to make a predicted mask on the color image
            
            raw_prediction = np.copy(prediction)
            thresh_prediction = threshold(prediction,.2) #threshold the prediction
            depth_pred = depth_prediction(depth_image,.96) #not needed, probably can be deleted
            x,r,grad,depth_pred1 = tp.plane_fit(thresh_prediction,depth_pred,depth_image) #plane_fit will be explained in toolpath.py
            
            x,y,cnt,bounding_cnt,box = tp.contours(thresh_prediction) #determines if its a square or a circle, and gets contour of shape and bounding shape
            
            scale_cnt = tp.scale_contour(cnt,bounding_cnt,.6) #smaller version of the contour
            scale_bounding_cnt = tp.scale_contour(bounding_cnt,bounding_cnt,.6) #smaller version of the bounding contour
            
            color_image,fitted_diff = tp.draw(color_image,x,y,cnt,bounding_cnt) #draw the contours on the image
            color_image,_ = tp.draw(color_image,x,y,scale_cnt,scale_bounding_cnt)
            #color_image = ar.detect_markers(color_image)
            
            
            
            cntbtwn = tp.interpolate_inc(scale_cnt,scale_bounding_cnt,.3) #get intermediate shapes
            cntbtwn2 = tp.interpolate_inc(cntbtwn,scale_bounding_cnt,0)
            
            color_image = cv2.drawContours(color_image,[cntbtwn],0,(150,0,0),3) #draw intermediate shapes
            color_image = cv2.drawContours(color_image,[cntbtwn2],0,(50,0,0),3)
            
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
            
            render(color_image=color_image,fitted_diff=fitted_diff,depth_colormap=depth_colormap,
                   raw_prediction=raw_prediction,thresh_prediction=thresh_prediction,depth_prediction=depth_pred1)
            
            centroid = pix2real(x,y,grad)
            xc = centroid[0][0]
            yc = centroid[0][1]
            
            
            
            key = cv2.waitKey(1)
            
            try: #self explanitory, play around with pressing certain keys to change exposure, white balance, and gain
                if key & 0xFF == ord('y'): 
                    expo += 300
                    print("Exposure:",expo)
                    color_sensor.set_option(rs.option.exposure, expo)
                if key & 0xFF == ord('u'):
                    expo -= 300
                    print("Exposure:",expo)
                    color_sensor.set_option(rs.option.exposure, expo)
                if key & 0xFF == ord('i'):
                    wb += 300
                    print("White balance:",wb)
                    color_sensor.set_option(rs.option.white_balance, wb)
                if key & 0xFF == ord('o'):
                    wb -= 300
                    print("White balance:",wb)
                    color_sensor.set_option(rs.option.white_balance, wb)
                if key & 0xFF == ord('k'):
                    gain += 1
                    print("gain:",2**gain)
                    color_sensor.set_option(rs.option.gain, 2**gain)
                if key & 0xFF == ord('l'):
                    gain -= 1
                    print("gain:",2**gain)
                    color_sensor.set_option(rs.option.gain, 2**gain)
            except:
                print("ERROR: Invalid camera parameter setting")
            # Press 'p' to print the toolpath to console and send to arduino
            if key & 0xFF == ord('t'):
                #toolpath1,toolpath_approx1 = tp.generate(scale_cnt,grad)
                toolpath2,toolpath_approx2 = tp.generate(cntbtwn,grad) #generate toolpath for cntbtwn
                temp = np.vstack(([250,250,108],toolpath_approx2))
                toolpath3,toolpath_approx3 = tp.generate(cntbtwn2,grad) #generate toolpath for cntbtwn2
                temp1 = np.vstack((temp,toolpath_approx3))
                toolpath4,toolpath_approx4 = tp.generate(scale_bounding_cnt,grad) #generate toolpath for bounding toolpath
                toolpath = np.vstack((temp1,toolpath_approx4))
                #toolpath_approx = tp.smooth(toolpath,5)
                #probe_toolpath = pb.generate(x,y,cnt,bounding_cnt,depth_image)
                tp.visualize(toolpath,toolpath) 
                #tp.visualize(probe_toolpath,probe_toolpath)
                #circle_toolpath = circ_points(x,y,radius-20)
                
                if USE_ARDUINO:
                    
                    #Trimmer.write_gcode("G00",X=200,Y=200,Z=108)
                    #Trimmer.write_gcode('M03')
                    Trimmer.write_np_to_arduino(toolpath)  #send toolpath to arduino, and therefore move gantry
                    #Trimmer.update()
                    Trimmer.write_gcode("P01") 
                    while(True):
                        Trimmer.update()
                        if Trimmer.trimmer_flag == "report":
                            Trimmer.trimmer_flag == ''
                            break
                    #Trimmer.write_gcode('M05')
                    Trimmer.write_gcode('G28') #home
                    #Trimmer.update()
                    #write_np_to_arduino(toolpath_approx,port='COM3')
                time.sleep(1)
                key = 1
            
            
            if key & 0xFF == ord('p'):
               test_data = pb.run_probe(Trimmer,[xc,yc,108],1,275) #run the probe
               saved = np.copy(test_data)
               stuff = pb.edge(test_data[0],.9) #analyse the probe data
               pb.visualize(test_data[0]) #visualize the probe data
               test_data[0][::,2] = test_data[0][::,3]
               test_data[0][::,0] -= (60 + 75)
               test_data[0][::,1] += 75
               #tp.visualize(test_data[0][::,:3],test_data[0][::,:3])
               
               
               
               _,toolpath = tp.generate(cnt,grad)
               tp.visualize(toolpath,test_data[0][::,:3])
               
               #print(stuff)
               
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            
                
    finally:
        if USE_ARDUINO:
            Trimmer.close()
        pipeline.stop()
        cv2.destroyAllWindows()
