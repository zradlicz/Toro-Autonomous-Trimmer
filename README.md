# Toro-Autonomous-Trimmer
Welcome to the Toro Autonomous Trimmer Project. This document will help you get started understanding the software behind this machine.
## Folder Layout
Folder | Contents
------------ | -------------
Calibration | code used to calibrate pixel space to gantry space
Gantry | code uploaded onto arduino
Live Detection | code needed to run the machine
Segmentation | code for creating NN model, as well as deprecated models
## Calibration
  The calibration folder contains the code used to automatically calibrate the machine. We have preformed this calibration for the current camera position, however if you make any modifications to the positioning of things on the robot it will likely have to be recalibrated. Offsets for different tools are currently defined in their respective python files. (Trimmer offset is defined in pix2real function in AutoTrim, probe offset is defined in probe.py)
## Gantry
  The gantry folder contains the arduino code that is currently uploaded onto the arduino.
### Controls
  The most straight forward way to communicate with the gantry is through the arduinos serial monitor. Here you will receive feedback from the gantry, and be able to send it numerous commands.

G-Code Command | Function
------------ | -------------
G28 | send gantry to home position
G00 X*int* Y<*int* Z*int* | send gantry to location x:*int* y:*int* z:<*int* (x and y range: 0-500, z range 0-108)
M03 | start trimmer motor spinning clockwise
M04 | start trimmer motor spinning counter-clockwise
M05 | stop trimmer motor from spinning
S*int* | set motor speed as PWM pulse (0-255)
F*int* | set feedrate of lead screws (0-50 mm/sec)
R00 | stop recording probe data
R01 | reset recording
R02 | start recording probe data
P01 | print "report\n"
???? | any unrecognized string will prompt the arduino to respond with "waiting\n" if it is waiting for a g-code command

## Live Detection
  Live detection is the most important folder. It contains the six python files needed to fully run the machine. AutoTrim is the main file, and the five other files are meant to be imported into AutoTrim, not run on their own. A breif description of each file can be found here.
### AutoTrim
  AutoTrim is the 'main' file of the machine. It is the program that you will run in order to control and get feedback from the machine. It draws from all of the files shown below to function properly. Functions that relate to the machine learning process are found in this file. Also, the main logic of the machine is found here.
![Output](/images/example_output.JPG)
This image is an example of the output you should get from running AutoTrim. From top left: RGB image, Depth image, shape difference, prediction, thresholded prediction, depth prediciton.
#### Controls
Key | Function
------------ | -------------
T | run the trimmer functionality
P | run the probe functinoality
Click | make the gantry go to the location you clicked
Y | increase exposure
U | decrease exposure
I | increase white balance
O | decrease white balance
K | increase gain
L | decrease gain
Q | quit and close all windows
ESC | quit and close all windows

### arduino
arduino is the file that allows python to communicate with the onboard arduino. 
### aruco
aruco is a file that contains a lot of the functions used for calibration as mentioned in the calibration section. It contains functions that AutoTrim uses to convert pixel values into realspace values, which can then be fed to the arduino.
### camera
camera contains all of the start up functions for the Intel RealSense camera. This includes depth alignment, resolution settings, bag file recordings and more.
### probe
probe contains the functions used to run the probe and get feedback from it, as well as functions to get gantry space information from depth maps.

![Probe Toolpath](/images/probe_toolpath.JPG)

This image shows the data collected from the probe overlayed onto the shape of the segmented shape.

![Probe Analysis](/images/probe_analyzed.JPG)

This image shows the edge detection working on the probe data.

### toolpath
toolpath contains all the functions that are used to generate and visualize the trimmer toolpath.
![Trimmer Toolpath](/images/toolpath.JPG)

## Segmentation

To use the segmentation code, get familiar with TensorFlow using Sentdex Youtube channel.

[This link](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) is what was used to creat the masks. Upload your additional images and use the polyline drawing tool to create maks for each image. Then export the .json file using the annotation tab on the top of the website. This json file and a folder of images will be used to generate the masks in maskGen.py


