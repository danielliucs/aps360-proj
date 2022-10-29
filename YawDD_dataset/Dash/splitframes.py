import cv2
import numpy as np
import csv
import os
import random
import torch
import torch.utils.data

img_hm_dashw = {} #dash women hashmap
img_hm_dashm = {} #dash male hashmap
videonum = 0
dash_female = ["1-FemaleNoGlasses.avi", "2-FemaleNoGlasses.avi", "3-FemaleGlasses.avi","4-FemaleNoGlasses.avi", 
"5-FemaleNoGlasses.avi", "6-FemaleNoGlasses.avi", "7-FemaleNoGlasses.avi", "8-FemaleGlasses.avi", "9-FemaleNoGlasses.avi",
"10-FemaleNoGlasses.avi", "11-FemaleGlasses.avi", "12-FemaleGlasses.avi", "13-FemaleGlasses.avi"] 

dash_male = ["1-MaleGlasses.avi", "2-MaleGlasses.avi", "3-MaleGlasses.avi", "4-MaleGlasses.avi", "5-MaleGlasses.avi", 
"6-MaleGlasses.avi", "7-MaleGlasses.avi", "8-MaleGlasses.avi", "9-MaleGlasses.avi", "10-MaleGlasses.avi",
"11-MaleGlasses.avi", "12-MaleGlasses.avi","13-MaleGlasses.avi", "14-MaleGlasses.avi", "15-MaleGlasses.avi", 
"16-MaleGlasses.avi"]



n = 1 # n is the number of images, its used for the op file path

# this function will do all the work to store the op files
def getFrames(time, vid, path):
    #vid.set(cv2.CAP_PROP_POS_MSEC, time*1000)
    frame_img = None
    success = None
    vid.set(cv2.CAP_PROP_POS_MSEC, time*1000)
    success, frame_img = vid.read()
    if not os.path.isdir(path):
        os.makedirs(path)

    if success:
        cv2.imwrite(path+"image"+str(n)+".jpg", frame_img)

    return (frame_img, success)

# this loop will send all the files to the getFrames function, and do all the frame rate work.
# loops through the input video, appends images to img_hm_dashw hashmap, and stores in FemaleFrames folder
while (videonum< len(dash_female)):

    #CHANGE THESE PATHS UP UNTIL "Female" TO BE YOUR LOCAL PATH.
    oppath = "/Users/srinidhishankar/Documents/APS360/aps360-proj/YawDD_dataset/Dash/Female/FemaleFrames/"
    path = "/Users/srinidhishankar/Documents/APS360/aps360-proj/YawDD_dataset/Dash/Female/" + str(dash_female[videonum])
    vid = cv2.VideoCapture(path)
    framerate = 30
    frameduration = round(1/framerate, 6)

    sec = 0
    
    img1_array = []
    frame_extract, success1 = getFrames(sec, vid, oppath)

    #this loop will loop through the video and store the frames
    while success1:
        n = n + 1
        sec = sec + frameduration
        sec = round(sec, 2)

        frame_extract, success1 = getFrames(sec, vid, oppath)
        img1_array.append(frame_extract)

    n = n+1
    img_hm_dashw[videonum] = img1_array
    videonum = videonum+1

videonum = 0

# this loop will send all the files to the getFrames function, and do all the frame rate work.
# loops through the input video, appends images to img_hm_dashm hashmap, and stores in MaleFrames folder
while (videonum< len(dash_male)):

    #CHANGE THESE PATHS UP UNTIL "Female" TO BE YOUR LOCAL PATH.
    oppath = "/Users/srinidhishankar/Documents/APS360/aps360-proj/YawDD_dataset/Dash/Male/MaleFrames/"
    path = "/Users/srinidhishankar/Documents/APS360/aps360-proj/YawDD_dataset/Dash/Male/" + str(dash_male[videonum])
    vid = cv2.VideoCapture(path)
    framerate = 30
    frameduration = round(1/framerate, 6)

    sec = 0
    
    img1_array = []
    frame_extract, success1 = getFrames(sec, vid, oppath)

    #this loop will loop through the video and store the frames
    while success1:
        n = n + 1
        sec = sec + frameduration
        sec = round(sec, 2)

        frame_extract, success1 = getFrames(sec, vid, oppath)
        img1_array.append(frame_extract)

    n = n+1
    img_hm_dashm[videonum] = img1_array
    videonum = videonum+1

