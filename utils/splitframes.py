from genericpath import isdir
import cv2
import numpy as np
import csv
import os
import random
import torch
import torch.utils.data

img_hm = {}
videonum = 0
all_file_paths = [] #paths to every video. don't change this, only the two paths underneath

#CHANGE THESE PATHS UP UNTIL "YawDD_dataset" TO BE YOUR LOCAL PATH.
directory = '/Users/srinidhishankar/Documents/APS360/aps360-proj/YawDD_dataset'
oppath = "/Users/srinidhishankar/Documents/APS360/aps360-proj/YawDD_dataset/allPhotos"


all_file_names = []
# iterate over files in
# that directory


#this fxn will go through the YawDD dataset and folders to save the file paths and names 
def scanFiles(directory):
    for filename in os.scandir(directory):
        if filename.is_dir():
            scanFiles(filename.path)
        if filename.is_file():
            all_file_paths.append(filename.path)
            all_file_names.append(filename.name)

n = 1 # n is the number of images, its used for the op file path

scanFiles(directory)

# this function will do all the work to store the op files
def getFrames(time, vid):
    #vid.set(cv2.CAP_PROP_POS_MSEC, time*1000)
    frame_img = None
    success = None
    vid.set(cv2.CAP_PROP_POS_MSEC, time*1000)
    success, frame_img = vid.read()
    #makes dir if not made 
    if not os.path.isdir(oppath):
        os.makedirs(oppath)

    if success:
        cv2.imwrite(oppath+"/"+all_file_names[videonum]+str(n)+".jpg", frame_img)

    return (frame_img, success)

# this loop will send all the files to the getFrames function, and do all the frame rate work.
# loops through the input video, appends images to img_hm_dashw hashmap, and stores in FemaleFrames folder
while (videonum< len(all_file_paths)):

    
    path = all_file_paths[videonum]
    vid = cv2.VideoCapture(path)

    #change these numbes (fps lower for less pics) if needed/ if model taking too long to train
    framerate = 3
    frameduration = round(1/framerate, 6)

    sec = 0
    
    img1_array = []
    frame_extract, success1 = getFrames(sec, vid)

    #this loop will loop through the video and store the frames
    while success1:
        n = n + 1
        sec = sec + frameduration
        sec = round(sec, 2)

        frame_extract, success1 = getFrames(sec, vid)
        img1_array.append(frame_extract)

    n = n+1
    img_hm[videonum] = img1_array
    videonum = videonum+1

