import cv2
from random import randrange

#Pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
#img = cv2.imread('girlface.jpg')
#or upload video
#webcam = cv2.VideoCapture('example.mp4')

#capture video from webcam.
webcam = cv2.VideoCapture(0)#can't work with IOS yet

#iterate forever over frames
while True:
  #read the current frame
  successful_frame_read, frame = webcam.read()
  
  #must convert to grayscale
  grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #Detect Faces
  face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

  #draw rectangles to show detection on the face_coordinates
  for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 3)
    
  cv2.imshow('Clever Programmer Face Detector', frame)
  key = cv2.waitKey(1)
    
    #stops if you hit Q
  if key==81 or key==113:
    break

#Releases the VideoCapture object
webcam.release()
