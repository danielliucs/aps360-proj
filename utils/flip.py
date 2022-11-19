import cv2
import os


def flip(dir, name):
    img_name = f'{dir}/{name}'
    image = cv2.imread(img_name)
    flippedimage = cv2.flip(image, 1)
    cv2.imwrite(img_name, flippedimage)
if __name__ == '__main__':
    dir = "C:/Users/User/Desktop/aps360-proj/testing_dataset/testing_normal"
    #all the videos before this point were fine
    #The first video that wasn't fine was this one
    first_vid = "IMG_8998_NormalMOV184.jpg"
    counter = 0
    for file in os.listdir(dir):
        if file != first_vid:
            counter += 1
        else:
            break

    new_counter = 0
    for file in os.listdir(dir):
        if new_counter == counter:
            flip(dir, file)
            continue
        new_counter += 1

