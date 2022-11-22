import cv2
import os
from PIL import Image

def crop_face(inpath, outpath): 
  
    image = Image.open(inpath)
    print(image.size)
    width, height = image.size   # Get dimensions
    
    left = (width - 640*2)/2
    top = (height - 480*2)/2
    right = (width + 640*2)/2
    bottom = (height + 480*2)/2

    cropped_image = image.crop((left, top, right, bottom))
    cropped_image.save(outpath)

# Crop the center of the image
def crop_folder(dir):
    for file in os.listdir(dir):
        crop_face(dir + "/" + file, dir + "/" + file)

crop_folder("C:/Users/leonz/github/aps360-proj/testing_dataset/testing_yawning")