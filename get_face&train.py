import cv2
#import sys
import logging as log
import datetime as dt
from PIL import Image
import numpy as np
import os
#from time import sleep


Path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(Path)   
face_id = input('\n Please enter the User id of the person: ')
print("\n Initializing face capture. Look the camera and please stay still for a few seconds ...")

count = 0# Initialize individual sampling face count
cam= cv2.VideoCapture(0)
anterior = 0

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
        
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for emergency
    if k == 27:
        break
    elif count >= 100: # Take 100 face sample and stop video
         break
# When everything is done, release the camera
cam.release()
cv2.destroyAllWindows()


datapath = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()    #LBPH=>local binary pattern histogram
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
# function to get the images 
def getImagesAndLabels(datapath):
    imagePaths = [os.path.join(datapath,f) for f in os.listdir(datapath)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8') #create a table with the values of the image

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(datapath)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml [create configuratio file]
recognizer.write('trainer/trainer.yml') 

# Print the number of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))