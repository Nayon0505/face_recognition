import face_recognition as fr
import cv2
import os
import pickle
print(cv2.__version__)

Encodings=[]
Names= []

with open('train.pkl', 'rb') as f:
    Names=pickle.load(f)
    Encodings= pickle.load(f)

font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)

while True:
    _,frame = cam.read()
    frameRGB= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    facePositions= fr.face_locations(frameRGB, model= 'cnn') #cnn ist ein besseres Model als das Standardmäßige HOG (Simpler und langsamer), da wir den Jetson Nano haben können wir cnn nutzen
    allEncodings = fr.face_encodings(frameRGB, facePositions)
    for(top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
        name='Unkown Person'
        matches = fr.compare_faces(Encodings, face_encoding)
        if True in matches:
            first_match_index= matches.index(True)
            name=Names[first_match_index]
        cv2.rectangle(frame,(left, top), (bottom, right), (0,0,255), 2)
        cv2.putText(frame,name,(left,top-6),font, .75, (0,255,255),2)
    cv2.imshow('Picture', frame)
    cv2.moveWindow('Picture', (0,0))
    if cv2.waitKey(0)== ord('q'):
        break
cam.release()
cv2.destroyAllWindows()