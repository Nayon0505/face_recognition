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
#cam = cv2.VideoCapture('/dev/video6', cv2.CAP_V4L2) #Mit Droidcam übers handy
cam = cv2.VideoCapture(0) #Mit Webcam

while True:
    _,frame = cam.read()
    frameSmall = cv2.resize(frame,(0,0), fx=.33,fy=.33) # Frame kleiner machen für besser Leistung
    frameRGB= cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)
    facePositions= fr.face_locations(frameRGB, model= 'cnn') #cnn ist ein besseres Model als das Standardmäßige HOG (Simpler und langsamer), da wir den Jetson Nano haben können wir cnn nutzen
    allEncodings = fr.face_encodings(frameRGB, facePositions)
    for(top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
        name='Unkown Person'
        matches = fr.compare_faces(Encodings, face_encoding)
        if True in matches:
            first_match_index= matches.index(True)
            name=Names[first_match_index]
        top=top*3
        right=right*3
        bottom=bottom*3
        left=left*3

        cv2.rectangle(frame,(left, top), (right, bottom), (0,0,255), 2) #Left right in Opnecv ist die X achse und Top Bottom Y, in facerecognition ist das anders, deshalb können die Kästen falsch gezeichnet werden, wenn man es falsch macht
        cv2.putText(frame,name,(left,top-6),font, .75, (0,255,255),2)
    cv2.imshow('Picture', frame)
    cv2.moveWindow('Picture', 0,0)
    if cv2.waitKey(1)== ord('q'):
        break
cam.release()
cv2.destroyAllWindows()