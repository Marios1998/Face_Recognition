import cv2
import logging as log
import datetime as dt

Path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(Path)
log.basicConfig(filename='webcam.log',level=log.INFO)
cam= cv2.VideoCapture(0)
anterior = 0

while True:
    if not cam.isOpened():
        print('App cannot use the camera right now')
        pass
    #get frame by frame
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #covnert to greyscale
    faces = faceCascade.detectMultiScale(     #detectMultScale returns a rectangle around the detected object input: 
        gray,
        scaleFactor=1.1,   #specified how much the image sizee is reduced with each scale
        minNeighbors=5,    #specifies how many neighbours each candidate rectagle should have to retainn it
        minSize=(30, 30)
    )
    #Detect the face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
    
    cv2.imshow('Video', frame)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    
    cv2.imshow('Video', frame)
    
    

