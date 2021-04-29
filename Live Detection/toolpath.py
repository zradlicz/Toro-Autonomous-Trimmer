# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:03:56 2021

@author: zradlicz
"""
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

import camera as cm
import AutoTrimV3 as ls





def contours(img):
    cent_img = img*255.0
    cent_img = cent_img.astype('uint8')
    cnts,hierarchy=cv2.findContours(cent_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    #print(cnt)
    
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX,cY = 0,0
    x = cX
    y = cY
    
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = box_cnt(box)
    #print(box)
    
    (xcirc,ycirc),radius = cv2.minEnclosingCircle(cnt)
    circle = circ_cnt(xcirc,ycirc,radius)
    #print(circle)
    
    shapetocircle = cv2.matchShapes(cnt,circle,1,0.0)
    shapetobox = cv2.matchShapes(cnt,box,1,0.0)
    if(shapetocircle<=shapetobox):
        bounding_cnt = circle
        shape = False
    else:
        bounding_cnt = box
        shape = True
        
    return x,y,cnt,bounding_cnt,shape

def circ_cnt(xc,yc,r):
    black_image = np.zeros((cm.cresolution()[1],cm.cresolution()[0]))
    black_image = cv2.circle(black_image,(int(xc),int(yc)),int(r),(255,0,0),-1)
    black_image = black_image.astype('uint8')

    cnts,hierarchy=cv2.findContours(black_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    circle_cnt = max(cnts, key=cv2.contourArea)
    
    return circle_cnt

def box_cnt(box):
    black_image = np.zeros((cm.cresolution()[1],cm.cresolution()[0]))
    #black_image = cv2.rectangle(black_image,(int(xc),int(yc)),int(r),(255,0,0),-1)
    #print(box[0])
    #print(box[1])
    black_image = cv2.drawContours(black_image,[box],0,(255,0,0),-1)
    black_image = black_image.astype('uint8')

    cnts,hierarchy=cv2.findContours(black_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    box_cnt = max(cnts, key=cv2.contourArea)
    
    return box_cnt
def ellipse_cnt(ellipse):
    black_image = np.zeros([cm.cresolution()[1],cm.cresolution()[0],3],dtype=np.uint8)
    black_image = cv2.circle(black_image,ellipse,(255,0,0),-1)
    black_image = black_image.astype('uint8')
    
    cnts,hierarchy=cv2.findContours(black_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ellipse_cnt = max(cnts, key=cv2.contourArea)
    
    return ellipse_cnt

def scale_contour(cnt, bounding_cnt, scale):
    Mc = cv2.moments(cnt)
    cxc = int(Mc['m10']/Mc['m00'])
    cyc = int(Mc['m01']/Mc['m00'])
    
    M = cv2.moments(bounding_cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cxc, cyc]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop


def plane_fit(pred,pred1,depth_image):
    xc,yc,cnt,bounding_cnt,_ = contours(pred)
    #x1,y1,cnt1,bounding_cnt1=contours(pred1)
    rect = cv2.boundingRect(bounding_cnt)
    srect = cv2.boundingRect(scale_contour(bounding_cnt,bounding_cnt,2))
    srect = np.asarray(srect)
    for i in range(len(srect)):
        if srect[i]<0:
            srect[i] = 0
            
    
    #rect = np.asarray(rect)
    #rect[0:2]=(rect[0:2]*.8).astype('int32')
    #rect[2:]=(rect[2:]*1.2).astype('int32')
    #rect1 = cv2.boundingRect(cnt1)
    #box = cv2.boxPoints(rect)
    #box = np.int0(box)
    
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #plt.imshow(depth_pred)
    spred = np.copy(pred)
    pred = pred[rect[1]:rect[1]+rect[2],rect[0]:rect[0]+rect[3]]
    spred = spred[srect[1]:srect[1]+srect[2],srect[0]:srect[0]+srect[3]]
    #pred1 = pred1[rect1[1]:rect1[1]+rect1[2],rect1[0]:rect1[0]+rect1[3]]
    depth = depth_image[rect[1]:rect[1]+rect[2],rect[0]:rect[0]+rect[3]]
    sdepth = depth_image[srect[1]:srect[1]+srect[2],srect[0]:srect[0]+srect[3]]
    #depth1 = depth_image[rect1[1]:rect1[1]+rect1[2],rect1[0]:rect1[0]+rect1[3]]
    i,j = np.where(spred>0)
    #i1,j1 = np.where(pred1>0)
    one = np.ones(len(i))
    #one1 = np.ones(len(i1))
    test = np.vstack((i,j))
    test = np.vstack((test,one)).T
    #test1 = np.vstack((i1,j1))
    #test1 = np.vstack((test1,one1)).T
    x,r,_,_ = np.linalg.lstsq(test,sdepth[i,j].T,rcond=None)
    avg = np.mean(sdepth[i,j])
    #x1,r1,_,_ = np.linalg.lstsq(test1,depth1[i1,j1].T,rcond=None)
    
    grad = np.zeros(sdepth.shape)
    for i in range(sdepth.shape[0]):
        for j in range(sdepth.shape[1]):
            grad[i][j] = i*x[0]+j*x[1]+x[2]
    
    error = avg*np.ones(depth_image.shape)
    error[srect[1]:srect[1]+srect[2],srect[0]:srect[0]+srect[3]] = abs(sdepth-grad)
    #error = abs(depth_image-grad)
    
    gradgrad = avg*np.ones(depth_image.shape)
    #gradgrad[rect[1]:rect[1]+rect[2],rect[0]:rect[0]+rect[3]] = grad
    thresh=3
    thresh_pred = np.copy(error)
    thresh_pred[error>=thresh] = 0
    thresh_pred[error<thresh] = 1
    
    #ax.scatter(i,j,sdepth[i,j])
    #ax.plot_trisurf(i,j,i*x[0]+j*x[1]+x[2])
    #ax.plot_trisurf(i1,j1,i1*x1[0]+j1*x1[1]+x1[2])
    #print(r)
    #print(r1)
    #ax.plot(toolpath_approx[:,0],toolpath_approx[:,1],toolpath_approx[:,2],color='red')
    #plt.show
    return x,r,gradgrad,thresh_pred

def plane_pred(pred,pred1,depth_image):  
    x,r,_,_ = plane_fit(pred,pred1,depth_image)
    error = np.zeros(depth_image.shape)
    #i,j = np.where(pred>0)
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            zact = depth_image[i][j]
            zpred = i*x[0]+j*x[1]+x[2]
            error[i][j] = zact-zpred
            
    return error


def generate(cnt,depth_image):
    toolpath = ls.pix2real(0,0,depth_image)
    for val in cnt:
        x=val[0][0]
        y=val[0][1]
        #z = depth_image[y,x]*depth_scale
        toolpath = np.vstack((toolpath,ls.pix2real(x,y,depth_image)))
    
    toolpath_approx = np.copy(toolpath)
    toolpath_approx[1:] = smooth(toolpath[1:], 5)
    return toolpath[1:], toolpath_approx[1:]


def smooth(tp,window):
    arr = np.copy(tp)
    def movingaverage (values, window): # modified function from stackoverflow
        extended_values = np.concatenate([values,values[:window-1]])
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(extended_values, weights, 'valid')
        return sma
    for i in range(arr.shape[1]):
        arr[:,i] = movingaverage(arr[:,i],window)
    return arr

# assumes that tp has shape N x 3
def downsample(tp,every):
    return tp[::every] # this is using np slicing: tp[start:stop:step]


def draw(img,x,y,cnt,bounding_cnt):
    img = cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
    img = cv2.drawContours(img, [cnt], 0, (255, 0, 0), 3)
    
    
    if (ls.successful_trim(cnt,.005)):
        color = (0,255,0)
    else:
        color = (0,0,255)
    img = cv2.drawContours(img,[bounding_cnt],0,color,2)
    # calculate contour fit differences
    contour = np.zeros([cm.cresolution()[1],cm.cresolution()[0],3],dtype=np.uint8)
    contour = cv2.drawContours(contour, [cnt], 0, color, -1)
    fitted_shape = np.zeros(contour.shape,dtype=np.uint8)
    fitted_shape = cv2.drawContours(fitted_shape,[bounding_cnt],0,color,-1)
    fitted_diff = (contour != fitted_shape).astype(np.uint8)*255
    
    return img, fitted_diff

def visualize(toolpath,toolpath_approx):
    print(toolpath)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.axes.set_xlim3d(left=0, right=500) 
    ax.axes.set_ylim3d(bottom=0, top=500) 
    ax.axes.set_zlim3d(bottom=0, top=108) 
    ax.plot(toolpath[:,0],toolpath[:,1],toolpath[:,2])
    ax.plot(toolpath_approx[:,0],toolpath_approx[:,1],toolpath_approx[:,2],color='red')
    plt.show

def interpolate_dec(shape_cnt, circ_cnt, val):
    Ms = cv2.moments(shape_cnt)
    cxs = int(Ms['m10']/Ms['m00'])
    cys = int(Ms['m01']/Ms['m00'])
    
    Mc = cv2.moments(circ_cnt)
    cxc = int(Mc['m10']/Mc['m00'])
    cyc = int(Mc['m01']/Mc['m00'])
    
    
    
    v1 = shape_cnt.shape[0]
    v2 = circ_cnt.shape[0]
    #print(v1)
    #print(v2)
    d = math.gcd(v1,v2)
    if d>=20:
        ss = shape_cnt[::int(v1/d)].astype(np.int32)
        cs = circ_cnt[::int(v2/d)].astype(np.int32)
    else:
        if v1>v2:
            d = v1/v2
            ss = shape_cnt[::int(d)].astype(np.int32)
            cs = circ_cnt
            while(cs.shape[0]!=ss.shape[0]):
                #ss = ss[:-1]
                #print(ss.shape[0])
                ss = np.delete(ss,cs.shape[0]%ss.shape[0],0)
                #print(ss.shape[0])
                #if ss.shape[0]>0:
                    #index = random.randint(0, int(ss.shape[0]-1))
                #ss = np.delete(ss, index)
        elif v2>v1:
            d = v2/v1
            cs = shape_cnt[::int(d)].astype(np.int32)
            ss = shape_cnt
            while(cs.shape[0]!=ss.shape[0]):
                #cs = cs[:-1]
                #print('circle has more points')
                cs = np.delete(cs,cs.shape[0]%ss.shape[0],0)
                #print(cs.shape[0])
                #if cs.shape[0]>0:
                    #index = random.randint(0, int(cs.shape[0]-1))
                #cs = np.delete(cs, index)
        else:
            pass
        
        
    #print(ss.shape)
    #print(cs.shape)
    
    cnt_norm_s = ss - [cxs, cys]
    cnt_norm_c = cs - [cxc, cyc]
    
    cntbtwn = (cnt_norm_c+cnt_norm_s)/(2+val)
    cntbtwn = cntbtwn + [cxc, cyc]
    cntbtwn = cntbtwn.astype(np.int32)
    return cntbtwn
    

def interpolate_inc(shape_cnt, circ_cnt, val):
    Ms = cv2.moments(shape_cnt)
    cxs = int(Ms['m10']/Ms['m00'])
    cys = int(Ms['m01']/Ms['m00'])
    
    Mc = cv2.moments(circ_cnt)
    cxc = int(Mc['m10']/Mc['m00'])
    cyc = int(Mc['m01']/Mc['m00'])
    
    
    
    v1 = shape_cnt.shape[0]
    v2 = circ_cnt.shape[0]
    #print(v1)
    #print(v2)
    sl = shape_cnt
    cl = circ_cnt
   
    if v1>v2:
        while(cl.shape[0]!=sl.shape[0]):
            index = random.randint(0,cl.shape[0]-1)
            #index = cl.shape[0]%sl.shape[0]
            avg = (cl[index-1]+cl[index])/2
            cl = np.insert(cl,index,avg,0)
    elif v2>v1:
        while(cl.shape[0]!=sl.shape[0]):
            index = random.randint(0,sl.shape[0]-1)
            #index = cl.shape[0]%sl.shape[0]
            avg = (sl[index-1]+sl[index])/2
            sl = np.insert(sl,index,avg,0)
    else:
        pass
        
        
    #print(ss.shape)
    #print(cs.shape)
    
    cnt_norm_s = sl - [cxs, cys]
    cnt_norm_c = cl - [cxc, cyc]
    
    cntbtwn = (cnt_norm_c+cnt_norm_s)/(2+val)
    cntbtwn = cntbtwn + [cxs, cys]
    cntbtwn = cntbtwn.astype(np.int32)
    return cntbtwn


#def generate_probe(x,y,cnt,bounding_cnt,depth_image):
#    toolpath = ls.pix2real(x,y,depth_image)
#    scale = scale_contour(bounding_cnt,1.5)
#    (lx,ly) = tuple(scale[scale[:,:,0].argmin()][0])
#    (rx,ry) = tuple(scale[scale[:,:,0].argmax()][0])
#    (bx,by) = tuple(scale[scale[:,:,1].argmin()][0])
#    toolpath = np.vstack((toolpath,ls.pix2real(lx,ly,depth_image)))
#    toolpath = np.vstack((toolpath,ls.pix2real(x,y,depth_image)))
#    toolpath = np.vstack((toolpath,ls.pix2real(rx,ry,depth_image)))
#    toolpath = np.vstack((toolpath,ls.pix2real(x,y,depth_image)))
#    toolpath = np.vstack((toolpath,ls.pix2real(bx,by,depth_image)))
#    return toolpath