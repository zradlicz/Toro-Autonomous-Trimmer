# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:36:40 2021

@author: zradlicz
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sc
import numpy.fft

import AutoTrim as ls
import toolpath as tp

#t = np.load('t.npy')
#raw = np.load('raw.npy')
#filtered = np.load('filtered.npy')
#
#data = np.array([[157.74, 192.1 ,  88.  ,  29.  ],
#        [159.  , 192.1 ,  88.  ,  27.  ],
#        [160.24, 192.1 ,  88.  ,  28.  ],
#        [161.5 , 192.1 ,  88.  ,  28.  ],
#        [162.74, 192.1 ,  88.  ,  29.  ],
#        [164.  , 192.1 ,  88.  ,  27.  ],
#        [165.24, 192.1 ,  88.  ,  29.  ],
#        [166.46, 192.1 ,  88.  ,  31.  ],
#        [167.72, 192.1 ,  88.  ,  29.  ],
#        [168.96, 192.1 ,  88.  ,  30.  ],
#        [170.22, 192.1 ,  88.  ,  28.  ],
#        [171.46, 192.1 ,  88.  ,  29.  ],
#        [172.72, 192.1 ,  88.  ,  31.  ],
#        [173.96, 192.1 ,  88.  ,  30.  ],
#        [175.22, 192.1 ,  88.  ,  28.  ],
#        [176.46, 192.1 ,  88.  ,  29.  ],
#        [177.72, 192.1 ,  88.  ,  31.  ],
#        [178.96, 192.1 ,  88.  ,  30.  ],
#        [180.24, 192.1 ,  88.  ,  29.  ],
#        [181.52, 192.1 ,  88.  ,  26.  ],
#        [182.8 , 192.1 ,  88.  ,  26.  ],
#        [184.04, 192.1 ,  88.  ,  29.  ],
#        [185.32, 192.1 ,  88.  ,  28.  ],
#        [186.56, 192.1 ,  88.  ,  30.  ],
#        [187.82, 192.1 ,  88.  ,  30.  ],
#        [189.06, 192.1 ,  88.  ,  29.  ],
#        [190.34, 192.1 ,  88.  ,  30.  ],
#        [191.6 , 192.1 ,  88.  ,  28.  ],
#        [192.84, 192.1 ,  88.  ,  28.  ],
#        [194.1 , 192.1 ,  88.  ,  28.  ],
#        [195.34, 192.1 ,  88.  ,  28.  ],
#        [196.6 , 192.1 ,  88.  ,  29.  ],
#        [197.84, 192.1 ,  88.  ,  29.  ],
#        [199.1 , 192.1 ,  88.  ,  29.  ],
#        [200.34, 192.1 ,  88.  ,  30.  ],
#        [201.62, 192.1 ,  88.  ,  29.  ],
#        [202.86, 192.1 ,  88.  ,  28.  ],
#        [204.12, 192.1 ,  88.  ,  27.  ],
#        [205.36, 192.1 ,  88.  ,  28.  ],
#        [206.62, 192.1 ,  88.  ,  28.  ],
#        [207.84, 192.1 ,  88.  ,  28.  ],
#        [209.1 , 192.1 ,  88.  ,  28.  ],
#        [210.34, 192.1 ,  88.  ,  28.  ],
#        [211.6 , 192.1 ,  88.  ,  28.  ],
#        [212.82, 192.1 ,  88.  ,  29.  ],
#        [214.06, 192.1 ,  88.  ,  28.  ],
#        [215.32, 192.1 ,  88.  ,  29.  ],
#        [216.56, 192.1 ,  88.  ,  28.  ],
#        [217.8 , 192.1 ,  88.  ,  27.  ],
#        [219.04, 192.1 ,  88.  ,  27.  ],
#        [220.3 , 192.1 ,  88.  ,  27.  ],
#        [221.54, 192.1 ,  88.  ,  29.  ],
#        [222.8 , 192.1 ,  88.  ,  27.  ],
#        [224.04, 192.1 ,  88.  ,  28.  ],
#        [225.3 , 192.1 ,  88.  ,  27.  ],
#        [226.54, 192.1 ,  88.  ,  27.  ],
#        [227.76, 192.1 ,  88.  ,  28.  ],
#        [229.02, 192.1 ,  88.  ,  28.  ],
#        [230.26, 192.1 ,  88.  ,  27.  ],
#        [231.52, 192.1 ,  88.  ,  29.  ],
#        [232.74, 192.1 ,  88.  ,  28.  ],
#        [234.  , 192.1 ,  88.  ,  26.  ],
#        [235.24, 192.1 ,  88.  ,  26.  ],
#        [236.5 , 192.1 ,  88.  ,  30.  ],
#        [237.74, 192.1 ,  88.  ,  26.  ],
#        [239.  , 192.1 ,  88.  ,  28.  ],
#        [240.24, 192.1 ,  88.  ,  27.  ],
#        [241.5 , 192.1 ,  88.  ,  27.  ],
#        [242.74, 192.1 ,  88.  ,  25.  ],
#        [244.  , 192.1 ,  88.  ,  26.  ],
#        [245.24, 192.1 ,  88.  ,  27.  ],
#        [246.5 , 192.1 ,  88.  ,  26.  ],
#        [247.74, 192.1 ,  88.  ,  24.  ],
#        [249.  , 192.1 ,  88.  ,  25.  ],
#        [250.24, 192.1 ,  88.  ,  25.  ],
#        [251.5 , 192.1 ,  88.  ,  25.  ],
#        [252.74, 192.1 ,  88.  ,  25.  ],
#        [254.  , 192.1 ,  88.  ,  26.  ],
#        [255.24, 192.1 ,  88.  ,  26.  ],
#        [256.5 , 192.1 ,  88.  ,  25.  ],
#        [257.74, 192.1 ,  88.  ,  25.  ],
#        [259.  , 192.1 ,  88.  ,  26.  ],
#        [260.24, 192.1 ,  88.  ,  28.  ],
#        [261.5 , 192.1 ,  88.  ,  26.  ],
#        [262.74, 192.1 ,  88.  ,  25.  ],
#        [264.  , 192.1 ,  88.  ,  26.  ],
#        [265.26, 192.1 ,  88.  ,  25.  ],
#        [266.54, 192.1 ,  88.  ,  25.  ],
#        [267.82, 192.1 ,  88.  ,  27.  ],
#        [269.06, 192.1 ,  88.  ,  26.  ],
#        [270.34, 192.1 ,  88.  ,  28.  ],
#        [271.6 , 192.1 ,  88.  ,  28.  ],
#        [272.84, 192.1 ,  88.  ,  24.  ],
#        [274.06, 192.1 ,  88.  ,  28.  ],
#        [275.34, 192.1 ,  88.  ,  26.  ],
#        [276.6 , 192.1 ,  88.  ,  26.  ],
#        [277.84, 192.1 ,  88.  ,  25.  ],
#        [279.06, 192.1 ,  88.  ,  26.  ],
#        [280.32, 192.1 ,  88.  ,  26.  ],
#        [281.56, 192.1 ,  88.  ,  24.  ],
#        [282.82, 192.1 ,  88.  ,  25.  ],
#        [284.04, 192.1 ,  88.  ,  27.  ],
#        [285.32, 192.1 ,  88.  ,  26.  ],
#        [286.56, 192.1 ,  88.  ,  26.  ],
#        [287.82, 192.1 ,  88.  ,  27.  ],
#        [289.04, 192.1 ,  88.  ,  24.  ],
#        [290.3 , 192.1 ,  88.  ,  26.  ],
#        [291.54, 192.1 ,  88.  ,  26.  ],
#        [292.8 , 192.1 ,  88.  ,  25.  ],
#        [294.02, 192.1 ,  88.  ,  28.  ],
#        [295.26, 192.1 ,  88.  ,  26.  ],
#        [296.52, 192.1 ,  88.  ,  28.  ],
#        [297.76, 192.1 ,  88.  ,  27.  ],
#        [299.  , 192.1 ,  88.  ,  26.  ],
#        [300.24, 192.1 ,  88.  ,  27.  ],
#        [301.5 , 192.1 ,  88.  ,  24.  ],
#        [302.74, 192.1 ,  88.  ,  27.  ],
#        [304.  , 192.1 ,  88.  ,  27.  ],
#        [305.24, 192.1 ,  88.  ,  26.  ],
#        [306.52, 192.1 ,  88.  ,  25.  ],
#        [307.76, 192.1 ,  88.  ,  26.  ],
#        [309.02, 192.1 ,  88.  ,  26.  ],
#        [310.26, 192.1 ,  88.  ,  26.  ],
#        [311.52, 192.1 ,  88.  ,  27.  ],
#        [312.76, 192.1 ,  88.  ,  26.  ],
#        [314.02, 192.1 ,  88.  ,  25.  ],
#        [315.26, 192.1 ,  88.  ,  28.  ],
#        [316.52, 192.1 ,  88.  ,  25.  ],
#        [317.76, 192.1 ,  88.  ,  25.  ],
#        [319.02, 192.1 ,  88.  ,  25.  ],
#        [320.26, 192.1 ,  88.  ,  26.  ],
#        [321.52, 192.1 ,  88.  ,  28.  ],
#        [322.76, 192.1 ,  88.  ,  25.  ],
#        [324.02, 192.1 ,  88.  ,  25.  ],
#        [325.26, 192.1 ,  88.  ,  26.  ],
#        [326.52, 192.1 ,  88.  ,  26.  ],
#        [327.76, 192.1 ,  88.  ,  24.  ],
#        [329.02, 192.1 ,  88.  ,  24.  ],
#        [330.26, 192.1 ,  88.  ,  26.  ],
#        [331.52, 192.1 ,  88.  ,  30.  ],
#        [332.76, 192.1 ,  88.  ,  26.  ],
#        [334.02, 192.1 ,  88.  ,  25.  ],
#        [335.26, 192.1 ,  88.  ,  25.  ],
#        [336.52, 192.1 ,  88.  ,  25.  ],
#        [337.76, 192.1 ,  88.  ,  26.  ],
#        [339.02, 192.1 ,  88.  ,  27.  ],
#        [340.24, 192.1 ,  88.  ,  26.  ],
#        [341.5 , 192.1 ,  88.  ,  27.  ],
#        [342.74, 192.1 ,  88.  ,  25.  ],
#        [344.  , 192.1 ,  88.  ,  25.  ],
#        [345.22, 192.1 ,  88.  ,  27.  ],
#        [346.46, 192.1 ,  88.  ,  26.  ],
#        [347.72, 192.1 ,  88.  ,  26.  ],
#        [348.96, 192.1 ,  88.  ,  27.  ],
#        [350.22, 192.1 ,  88.  ,  27.  ],
#        [351.46, 192.1 ,  88.  ,  24.  ],
#        [352.72, 192.1 ,  88.  ,  27.  ],
#        [353.96, 192.1 ,  88.  ,  25.  ],
#        [355.2 , 192.1 ,  88.  ,  28.  ],
#        [356.44, 192.1 ,  88.  ,  25.  ],
#        [357.7 , 192.1 ,  88.  ,  25.  ],
#        [358.94, 192.1 ,  88.  ,  28.  ],
#        [360.16, 192.1 ,  88.  ,  27.  ],
#        [361.42, 192.1 ,  88.  ,  26.  ],
#        [362.66, 192.1 ,  88.  ,  24.  ],
#        [363.92, 192.1 ,  88.  ,  27.  ],
#        [365.14, 192.1 ,  88.  ,  26.  ],
#        [366.4 , 192.1 ,  88.  ,  25.  ],
#        [367.64, 192.1 ,  88.  ,  25.  ],
#        [368.9 , 192.1 ,  88.  ,  26.  ],
#        [370.14, 192.1 ,  88.  ,  25.  ],
#        [371.4 , 192.1 ,  88.  ,  25.  ],
#        [372.64, 192.1 ,  88.  ,  24.  ],
#        [373.9 , 192.1 ,  88.  ,  27.  ],
#        [375.14, 192.1 ,  88.  ,  25.  ],
#        [376.4 , 192.1 ,  88.  ,  26.  ],
#        [377.64, 192.1 ,  88.  ,  25.  ],
#        [378.9 , 192.1 ,  88.  ,  25.  ],
#        [380.14, 192.1 ,  88.  ,  24.  ],
#        [381.4 , 192.1 ,  88.  ,  25.  ],
#        [382.64, 192.1 ,  88.  ,  26.  ],
#        [383.9 , 192.1 ,  88.  ,  25.  ],
#        [385.14, 192.1 ,  88.  ,  25.  ],
#        [386.4 , 192.1 ,  88.  ,  26.  ],
#        [387.64, 192.1 ,  88.  ,  28.  ],
#        [388.9 , 192.1 ,  88.  ,  26.  ],
#        [390.14, 192.1 ,  88.  ,  26.  ],
#        [391.4 , 192.1 ,  88.  ,  26.  ],
#        [392.64, 192.1 ,  88.  ,  25.  ],
#        [393.9 , 192.1 ,  88.  ,  28.  ],
#        [395.14, 192.1 ,  88.  ,  28.  ],
#        [396.4 , 192.1 ,  88.  ,  27.  ],
#        [397.64, 192.1 ,  88.  ,  28.  ],
#        [398.9 , 192.1 ,  88.  ,  27.  ],
#        [400.14, 192.1 ,  88.  ,  25.  ],
#        [401.4 , 192.1 ,  88.  ,  27.  ],
#        [402.64, 192.1 ,  88.  ,  28.  ],
#        [403.9 , 192.1 ,  88.  ,  26.  ],
#        [405.14, 192.1 ,  88.  ,  29.  ],
#        [406.36, 192.1 ,  88.  ,  27.  ],
#        [407.62, 192.1 ,  88.  ,  28.  ],
#        [408.86, 192.1 ,  88.  ,  27.  ],
#        [410.12, 192.1 ,  88.  ,  28.  ],
#        [411.34, 192.1 ,  88.  ,  26.  ],
#        [412.6 , 192.1 ,  88.  ,  28.  ],
#        [413.84, 192.1 ,  88.  ,  27.  ],
#        [415.1 , 192.1 ,  88.  ,  26.  ],
#        [416.32, 192.1 ,  88.  ,  26.  ],
#        [417.56, 192.1 ,  88.  ,  26.  ],
#        [418.82, 192.1 ,  88.  ,  27.  ],
#        [420.06, 192.1 ,  88.  ,  27.  ],
#        [421.3 , 192.1 ,  88.  ,  28.  ],
#        [422.54, 192.1 ,  88.  ,  29.  ],
#        [423.8 , 192.1 ,  88.  ,  26.  ],
#        [425.04, 192.1 ,  88.  ,  27.  ],
#        [426.26, 192.1 ,  88.  ,  27.  ],
#        [427.52, 192.1 ,  88.  ,  28.  ],
#        [428.76, 192.1 ,  88.  ,  26.  ],
#        [430.02, 192.1 ,  88.  ,  27.  ],
#        [431.24, 192.1 ,  88.  ,  28.  ],
#        [432.5 , 192.1 ,  88.  ,  25.  ]])
#
#kernel = np.array([-1,0,1])
#
#raw = data[::,3]
#xloc = data[::,0]
#yloc = data[::,1]
#zloc = data[::,2]

def convolve(data,kernel):
    der = np.zeros(len(data))    
    for index in range(1,len(data)-1):
        der[index] = data[index-1]*kernel[0]+data[index]*kernel[1]+data[index+1]*kernel[2]
    return der
    
def movingaverage (values, window): # modified function from stackoverflow
    extended_values = np.concatenate([values,values[:window-1]])
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(extended_values, weights, 'valid')
    return sma

def edge(data,thresh):
    kernel = np.array([-1,0,1])
    raw = data[::,3]
    xloc = data[::,0]
    yloc = data[::,1]
    zloc = data[::,2]
    smooth = movingaverage(raw,20)
    deriv = convolve(smooth,kernel)
    indicies = np.array([]).astype('int32')
    for index in range(1,len(deriv)-1):
        if deriv[index]==np.max(deriv):
            indicies = np.append(indicies,index)
        elif deriv[index]==np.min(deriv):
            indicies = np.append(indicies,index)
            pass
        else:
            pass
    return indicies
            


def norm(data):
    avg = np.average(data)
    return data/avg

def generate(x,y,cnt,bounding_cnt,depth_image):
    toolpath = ls.pix2real(x,y,depth_image)
    scale = tp.scale_contour(bounding_cnt,bounding_cnt,1.3)
    (lx,ly) = tuple(scale[scale[:,:,0].argmin()][0])
    (rx,ry) = tuple(scale[scale[:,:,0].argmax()][0])
    (bx,by) = tuple(scale[scale[:,:,1].argmin()][0])
    toolpath = np.vstack((toolpath,ls.pix2real(lx,ly,depth_image)))
    toolpath = np.vstack((toolpath,ls.pix2real(x,y,depth_image)))
    toolpath = np.vstack((toolpath,ls.pix2real(rx,ry,depth_image)))
    toolpath = np.vstack((toolpath,ls.pix2real(x,y,depth_image)))
    toolpath = np.vstack((toolpath,ls.pix2real(bx,by,depth_image)))
    return toolpath


PROBE_LAG = 75 # from probe axis to wheel in mm
PROBE_OFFSET = np.array([60,75,0]) # from tool point to probe in mm
VERT_CLEARANCE = -20
    
    # initial_points = [cent - [PROBE_LAG,0,0] + [0,0,VERT_CLEARANCE],
    #           cent - [PROBE_LAG,0,0],
    #           cent]
    # ta.write_np_to_arduino(np.vstack(initial_points) - PROBE_OFFSET[None,:])
    
def run_probe_old(ta,cent,n_spokes,test_distance):
    test_data = []
    cent = np.asarray(cent)
    for i in range(n_spokes):
        theta = i * 2*np.pi/n_spokes
        direction = np.array([np.cos(theta), np.sin(theta),0])
        initial_points = [cent - PROBE_LAG * direction + [0,0,VERT_CLEARANCE],
              cent - PROBE_LAG * direction + [0,0,VERT_CLEARANCE],
              cent + [0,0,VERT_CLEARANCE]]
        ta.write_np_to_arduino(np.vstack(initial_points) - PROBE_OFFSET[None,:])

        next_point = cent + PROBE_LAG * direction + [0,0,VERT_CLEARANCE] - PROBE_OFFSET
        ta.write_gcode("G00",X=next_point[0],Y=next_point[1],Z=next_point[2])
        data_out = test_to(ta,cent + (PROBE_LAG + test_distance) * direction)
        test_data.append(data_out)
    
    return test_data

def run_probe(ta,cent,n_spokes,test_distance):
    test_data = []
    cent = np.asarray(cent)
    for i in range(n_spokes):
        theta = i * 2*np.pi/n_spokes
        direction = np.array([np.cos(theta), np.sin(theta),0])
        initial_points = [cent - PROBE_LAG * direction + [0,0,VERT_CLEARANCE],
              cent - PROBE_LAG * direction + [0,0,VERT_CLEARANCE],
              cent + [0,0,VERT_CLEARANCE]]
        ta.write_np_to_arduino(np.vstack(initial_points) - PROBE_OFFSET[None,:])

        next_point = cent + PROBE_LAG * direction + [0,0,VERT_CLEARANCE] - PROBE_OFFSET
        ta.write_gcode("G00",X=next_point[0],Y=next_point[1],Z=next_point[2])
        data_out = test_to(ta,cent + (PROBE_LAG + test_distance) * direction + [0,0,VERT_CLEARANCE] - PROBE_OFFSET)
        test_data.append(data_out)
    
    return test_data

def run_probe_new(ta,cent,n_spokes,test_distance):
    depth = 80
    test_data = []
    for i in range(n_spokes):
        dist = [test_distance,0,0]
        cent[2] = depth
        ta.write_np_to_arduino(cent+PROBE_OFFSET+dist)
        ta.write_np_to_arduino(cent+PROBE_OFFSET)
        ta.write_np_to_arduino(cent+PROBE_OFFSET-dist)
        data_out = test_to(ta,cent + (PROBE_LAG + test_distance) * direction)
        test_data.append(data_out)
    
    return test_data
        
    
def test_to(ta,point):
    point = np.round(point,decimals=3)
    ta.write_gcode("R01")
    ta.write_gcode("G00",X=point[0],Y=point[1])
    ta.write_gcode("R00")
    ta.write_gcode("P01")
    while True:
            ta.update()
            if ta.trimmer_flag == 'report':
                ta.trimmer_flag = ''
                data = np.array(ta.data)
                ta.data = []
                return data

def visualize(data):
    raw = data[::,3]
    xloc = data[::,0]
    yloc = data[::,1]
    zloc = data[::,2]
    kernel = np.array([-1,0,1])
    smooth = movingaverage(raw,20)
    filterednorm = norm(smooth)
    deriv = convolve(smooth,kernel)
    acc = convolve(deriv,kernel)
    derivsmooth = movingaverage(deriv,35)
    #t = np.linspace(0,300,2000)
    plt.plot(xloc,smooth)
    plt.scatter(xloc,derivsmooth)
    
    edge_indicies = edge(data,1)
    
    print(edge_indicies)
    for value in edge_indicies:
       plt.axvspan(xloc[value], xloc[value]+1.5, facecolor='b', alpha=0.5)
    


#if __name__ == "__main__":
#    import matplotlib.pyplot as plt
#    import trimmerarduino
#    #with trimmerarduino.TAtest('COM7',115200,1) as ta:
#    with trimmerarduino.TrimmerArduinoNoblock('COM7',115200,1) as ta:
#        ta.start_connection()
#        ta.home()
#        datas = run_probe(ta,np.array([200,200,100]),4,50)
#        for data in datas:
#            plt.figure()
#            plt.plot(data[:,0])
#            plt.plot(data[:,1])
#            plt.plot(data[:,2])
#            plt.figure()
#            plt.plot(data[:,3])


#smooth = movingaverage(raw,20)
#filterednorm = norm(smooth)
#filterednorm -=1
#deriv = convolve(smooth,kernel)
#acc = convolve(deriv,kernel)
#derivsmooth = movingaverage(deriv,35)
##t = np.linspace(0,300,2000)
#plt.plot(xloc,filterednorm)
#plt.scatter(xloc,derivsmooth)
#
#edge_indicies = edge(data,1)
#
#print(edge_indicies)
#for value in edge_indicies:
#    plt.axvspan(xloc[value], xloc[value]+1.5, facecolor='b', alpha=0.5)
    
    
#import numpy as np
#from scipy.fftpack import fft
#from scipy import signal
#N = 2000
#T = 1.0/1000.0
#x = np.linspace(0.0, N*T,N)
#a = np.sin(50.0 * 2.0*np.pi*x)
#b = 0.5*np.sin(80.0 * 2.0*np.pi*x)
#y = a + b
##y = raw-raw.mean()
#y = signal.detrend(raw)
#yf = fft(y)
#Y = N/2 * np.abs(yf[0:int(N/2)])
#X = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))
#
#plt.plot(X[:200],Y[:200])
#plt.grid()
#plt.show() 
