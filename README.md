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
## Gantry
## Live Detection
Live detection is the most important folder. It contains the six python files needed to fully run the machine. A breif description of each file can be found here.
### AutoTrim
AutoTrim is the 'main' file of the machine. It is the program that you will run in order to control and get feedback from the machine. It draws from all of the files shown below to function properly. Functions that relate to the machine learning process are found in this file. Also, the main logic of the machine is found here.
### arduino
arduino is the file that allows python to communicate with the onboard arduino. 
### aruco
aruco is a file that contains a lot of the functions used for calibration as mentioned in the calibration section. It contains functions that AutoTrim uses to convert pixel values into realspace values, which can then be fed to the arduino.
### camera
camera contains all of the start up functions for the Intel RealSense camera. This includes depth alignment, resolution settings, bag file recordings and more.
### probe
probe contains the functions used to run the probe and get feedback from it, as well as functions to get gantry space information from depth maps.
### toolpath
toolpath contains all the functions that are used to generate and visualize the trimmer toolpath. 

## Segmentation



