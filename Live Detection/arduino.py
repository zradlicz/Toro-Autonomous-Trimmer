# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:45:00 2021

@author: Alan

This file contains code for communicating with an Arduino running version 5 or 6
of the FancyCNC code written for our trimmer. Examples of how to use this code
can be found in the __main__ block at the end of this file.

NOTE: you may need to call start_connection a few times if it doesn't reliably
connect on the first try.
"""

import serial as sr
import numpy as np
import time
from collections import deque

# there are some differences in function names that could break this code if using
# pyserial version before 3.0
assert float(sr.__version__) >= 3

# this is the simple version which blocks until the gcode gets set to the Arduino
# this is also missing some features the Noblock version has
# make sure to call start_connection() at some point before sending gcode
class TrimmerArduino(sr.Serial):
    waiting = False
    
    def __init__(self,port,baudrate,**kwargs):
        super().__init__(port,baudrate,**kwargs)
    
    def start_connection(self):
        time.sleep(0.1)
        self.flushInput()
        self.flushOutput()
        time.sleep(0.1)
        self.write(b"?\n")
        
    def write_np_to_arduino(self,arr):
        arr = np.round(arr,decimals = 3) # having too many decimal places causes errors
        for row in arr:
            self.write_gcode("G00",X=row[0],Y=row[1],Z=row[2])
    
    def readline_arduino(self):
        line = self.readline()
        line = line.decode('UTF-8').strip("\r\n")
        if line == "waiting":
            self.waiting = True
        return line
    
    def block_until_waiting(self):
        while not self.waiting:
            print("Got:",self.readline_arduino())
        return
    
    def write_gcode(self,code,**kwargs):
        self.block_until_waiting()
        self.waiting = False
        command = b' '.join(
            [bytes(code,'UTF-8')] + 
            [bytes(k+str(v),'UTF-8') for k,v in kwargs.items()] +
            [bytes('\n','UTF-8')])
        print("Writing command:", command)
        print("Bytes written:", self.write(command))
    
    def home(self):
        self.write_gcode("G28")


# this is the simple version which keeps gcode in a queue and tries to send the
# stored gcode every time update() is called
# make sure to call start_connection() at some point before sending gcode, and
# put update() in a loop
class TrimmerArduinoNoblock(sr.Serial):
    waiting = False
    state_flag = ''
    trimmer_flag = ''
    data = []
    
    def __init__(self,port,baudrate,timeout,**kwargs):
        super().__init__(port,baudrate,timeout=timeout,**kwargs)
        self.tasks = deque()
    
    def start_connection(self):
        time.sleep(1)
        self.flushInput()
        self.flushOutput()
        time.sleep(0.1)
        print("Writing ?:")
        print(self.write(b"?\n"))
        time.sleep(1)
    
    def update(self,max_lines=15):
        for i in range(max_lines):
            line = self.readline_arduino()
            if len(line) == 0:
                break
            print("Got:", line)
        try:
            task = self.tasks.popleft()
        except IndexError:
            return
        
        if task[0] == 'write':
            if self.waiting:
                self.waiting = False
                print("Writing command:", task[1])
                print("Bytes written:", self.write(task[1]))
            else:
                self.tasks.appendleft(task)
        elif task[0] == 'flag':
            self.state_flag = task[1]
    
    def readline_arduino(self):
        if self.in_waiting:
            line = self.readline()
            line = line.decode('UTF-8').strip("\r\n")
            if line == "waiting":
                self.waiting = True
            if line == "report":
                self.trimmer_flag = "report"
            if line[:4] == "data":
                self.data.append(list(map(float,line[5:].split(','))))
            return line
        else:
            return ''
    
    def write_gcode(self,code,**kwargs):
        command = b' '.join(
            [bytes(code,'UTF-8')] + 
            [bytes(k+str(v),'UTF-8') for k,v in kwargs.items()] +
            [bytes('\n','UTF-8')])
        self.tasks.append( ('write',command) )
    
    def queue_flag(self,flag):
        self.tasks.append( ('flag',flag) )
    
    def get_flag(self):
        return self.state_flag
    
    def cancel_writes(self):
        self.tasks.clear()
    
    def write_np_to_arduino(self,arr):
        arr = np.round(arr,decimals = 3) # having too many decimal places causes errors
        for row in arr:
            self.write_gcode("G00",X=row[0],Y=row[1],Z=row[2])
    
    def home(self):
        self.write_gcode("G28")


# dummy class which acts like TrimmerArduinoNoblock but without a real connection
# to the Arduino. Any TrimmerArduinoNoblock can be directly replaced with TAtest
# for debugging purposes.
class TAtest(TrimmerArduinoNoblock):
    is_open = True
    
    def __init__(self,port,baudrate,timeout,**kwargs):
        pass
    
    def start_connection(self):
        self.waiting = True
        
    def write_np_to_arduino(self,arr):
        arr = np.round(arr,decimals = 3) # having too many decimal places causes errors
        for row in arr:
            self.write_gcode("G00",X=row[0],Y=row[1],Z=row[2])
    
    def write_gcode(self,code,**kwargs):
        command = ' '.join(
            [code] + 
            [k+str(v) for k,v in kwargs.items()] +
            ['\n'])
        print(command)
    
    def update(self):
        self.trimmer_flag = 'report'
        self.data = np.array([[1,2,3,20],
                              [2,3,1,25],
                              [3,4,-1,22]])
    
    def home(self):
        self.write_gcode("G28")
    
    def close(self):
        pass



if __name__ == "__main__":
    steps = 300
    t = np.linspace(0,3.14*3,num=steps);
    arr = np.zeros( (300,3) )
    arr[:,0] = 70*np.sin(t)+150
    arr[:,1] = 70*np.cos(t)+150
    arr[:,2] = 50*np.cos(4*t)+150
    arr = np.round(arr,decimals = 3)
    
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(arr[:,0],arr[:,1],arr[:,2])
    '''
    '''
    with TrimmerArduino('COM7',115200,timeout=1) as ta:
        ta.start_connection()
        ta.write_gcode("G00",X=50,Y=50,Z=150)
        ta.write_np_to_arduino(arr)
        ta.close()
    '''
    '''
    with TrimmerArduinoNoblock('COM7',115200,1) as ta:
        ta.start_connection()
        ta.home()
        ta.write_gcode("G00",X=50,Y=50,Z=150)
        ta.write_gcode("F50")
        ta.write_np_to_arduino(arr)
        ta.queue_flag('done')
        n_updates = 0
        while True:
            n_updates += 1
            ta.update()
            #print(n_updates)
            if ta.get_flag() == 'done':
                print('updates:',n_updates)
                break'''
    
    import matplotlib.pyplot as plt
    
    def movingaverage (values, window): # modified function from stackoverflow
        extended_values = np.concatenate([values,values[:window-1]])
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(extended_values, weights, 'valid')
        return sma
    
    with TrimmerArduinoNoblock('COM7',115200,1) as ta:
        ta.start_connection()
        ta.write_gcode("G00",Z=108)
        ta.home()
        ta.write_gcode("G00",X=200,Y=200,Z=100)
        ta.write_gcode("G00",X=200,Z=60)
        ta.write_gcode("R01")
        ta.write_gcode("G00",X=350)
        ta.write_gcode("R00")
        ta.write_gcode("P01")
        
        while True:
            ta.update()
            if ta.trimmer_flag == 'report':
                data = np.array(ta.data)
                plt.figure()
                plt.plot(data[1:,0])
                plt.plot(data[1:,1])
                plt.plot(data[1:,2])
                plt.figure()
                plt.plot(data[1:,3],'k.')
                plt.plot(movingaverage(data[1:,3],10))
                break