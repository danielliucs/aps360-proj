import face_recognition
import cv2
from PIL import Image
import torchvision.transforms as transforms

'''
crop_face
    attemps to take an image, specified by inpath, and detect a face.  If detected, it will crop the 
    image to only contain the face and write it to ann image specified by outpath.

    if a face is detected and cropped image produced, returns 0.  If no face detected, does not write
    any files and returns 1
'''
def crop_face(inpath, outpath): 
    #load the image
    img = face_recognition.load_image_file(inpath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #detect faces
    face_locations = face_recognition.face_locations(img_rgb)

    #error check to see if no faces were found
    if len(face_locations) == 0:
        print("no faces found in the photo.  Assuming no yawn.")
        return 1

    #cop the picture
    for top, right, bottom, left in face_locations:
        crop_img = img_rgb[top:bottom, left:right]
        cv2.imwrite(outpath, crop_img)

    return 0


'''
avg_RGB
    averages the color channels of an image tensor and returns avg as an int.
'''
def avg_RGB(cropped, x, y):
    ret = int((cropped[0, y, x].item() + cropped[1, y, x].item() + cropped[2, y, x].item())/3)
    return ret


'''
sample_for_yawn
    samples the mouth region of the image given by inpath, with region as specified in the proposal,
    and takes the average of the samples.  returns this average
'''
def sample_for_yawn(inpath): 
    #get the cropped face as a tensor
    cropped_face = Image.open(inpath) #_cropped
    transform = transforms.Compose([transforms.PILToTensor()])
    cropped_tensor = transform(cropped_face)

    #get pixel sample values
    anchx = int(cropped_tensor.shape[2]/2)      #haflway inbetween the face
    anchy = int(cropped_tensor.shape[1]*3/4)    #2/3 of the way down from the top (1/3 of the way up) in theory, but the cropping works better with 3/4

    #NOTES:  0 is B, 255 W.  Access tensors in the form of [C, Y, X].
    xoffset = int(cropped_tensor.shape[2]/16)    
    yoffset = int(cropped_tensor.shape[1]/16)

    #TAKE SAMPLES
    #center
    samp1 = avg_RGB(cropped_tensor, anchx, anchy)
    #left of center
    samp2 = avg_RGB(cropped_tensor, anchx-xoffset, anchy)
    #right of center
    samp3 = avg_RGB(cropped_tensor, anchx+xoffset, anchy)
    #below center
    samp4 = avg_RGB(cropped_tensor, anchx, anchy+yoffset)
    
    return ((samp1 + samp2 + samp3 + samp4)/4)

''' SIMPLE SINGLE IMAGE CLASSIFIER
#input an image name and file ext here
path = 'noyawn'
ext  = 'jpg'

err = crop_face(path + '.' + ext, path + '_cropped.' + ext)
if err == 1:
    print('no yawn detected (err default)')
    exit()

yawn_sample = sample_for_yawn(path + '_cropped.' + ext)

#predict a yawn
if(yawn_sample <= 255/2):
    print('yawn detected')
else:
    print('no yawn detected')
'''

#accuracy test for female no yawning
count = 0
total = 0
for i in range(103, 115):
    inpath  = '../dataset/Yawning/1-FemaleNoGlasses-Yawning.avi{0}.jpg'.format(i)
    outpath = '../dataset/Yawning/1-FemaleNoGlasses-Yawning.avi{0}_FACECROPPED.jpg'.format(i)
    
    err = crop_face(inpath, outpath)
    if err == 1:
        print('no yawn detected (err default) -', inpath)
        total += 1
        continue

    yawn_sample = sample_for_yawn(outpath)

    #predict a yawn
    if(yawn_sample <= 255/2):
        print('yawn detected -', inpath)
        count += 1
    else:
        print('no yawn detected -', inpath)

    total += 1

print("baseline accuracy: ", 100*count/total, "%", sep='')
