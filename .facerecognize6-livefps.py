import face_recognition as fr
import cv2
import os
import pickle
import time
print(cv2.__version__)

fpsReport=0 # FÜr die FPS ermittlung
scaleFactor=.25 # Wir verwenden den um änderungen im gesamten Model vorzunehmen um die FPSS zu optimieren
Encodings=[]
Names= []

########################################

with open('train.pkl', 'rb') as f:
    Names=pickle.load(f)
    Encodings= pickle.load(f)

font = cv2.FONT_HERSHEY_SIMPLEX
#cam = cv2.VideoCapture('/dev/video6', cv2.CAP_V4L2) #Mit Droidcam übers handy
cam = cv2.VideoCapture(0) #Mit Webcam
process_this_frame = True # Jeder n'te frame soll nur geprüft werden für Leistung

timeStamp = time.time() # Hier wird der Anfang des Loops festgehalten

fps = cam.get(cv2.CAP_PROP_FPS)
width = int(cam.get(3))
height = int(cam.get(4))

output = cv2.VideoWriter("assets/videos/rec.mp4",
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),  #Video speichern
            fps=fps, frameSize=(width, height))

#########################################
while True:
    _,frame = cam.read()

    if process_this_frame:
        frameSmall = cv2.resize(frame,(0,0), fx=scaleFactor,fy=scaleFactor)         # Frame kleiner machen für besser Leistung
        frameRGB= cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)
        facePositions= fr.face_locations(frameRGB, model= 'hog')                    #cnn ist ein besseres Model als das Standardmäßige HOG (Simpler und langsamer), da wir den Jetson Nano haben können wir cnn nutzen
        allEncodings = fr.face_encodings(frameRGB, facePositions)
        for(top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):
            name='Unbekannte Person'
            matches = fr.compare_faces(Encodings, face_encoding)
            if True in matches:
                first_match_index= matches.index(True)
                name=Names[first_match_index]
            top=int(top/scaleFactor)
            right=int(right/scaleFactor)
            bottom=int(bottom/scaleFactor)
            left=int(left/scaleFactor)

            cv2.rectangle(frame,(left, top), (right, bottom), (0,0,255), 2)             #Left right in Opnecv ist die X achse und Top Bottom Y, in facerecognition ist das anders, deshalb können die Kästen falsch gezeichnet werden, wenn man es falsch macht
            cv2.putText(frame,name,(left,top-6),font, .75, (0,255,255),2)
            dt=time.time() - timeStamp                                                      # Jetzt nehmen wir die jetzige Zeit - die Zeit am beginn des Loops damit wir die Differenz haben
            fps= 1/dt                                                                       # Wie viele Frames pro Sekunde, bsp: ein Frame jede 1/10 Sekunde, dann ist 1/(1/10)= 10 FPS
            fpsReport=.9*fpsReport + .1*fps  
                
    process_this_frame = not process_this_frame #Not ist wie ! in Java, er schaltet den Boolean Wert immer um, sodass nur jeder 2. Frame gecheckt wird
                                                   #Low pass filter, FPS sind die gemessene Frames, die ändern sich zu sehr und springen. Wir geben dem alten Report 95% Vertrauen und dem neuen 5 %. Bei zu großen Sprüngen, bleibt der Report trotdem konstant. 
                                                                                        # Wenn die FPS hochgehen, gehen sie immer nur um 5% hoch, also wenn sie konstant oben sind, ändert sich der Report gleichmäßog und kontinuierlich
    timeStamp= time.time()
    cv2.rectangle(frame,(0,0),(100,40),(0,0,255),-1)                                # -1 macht das Rechteck solid
    cv2.putText(frame, str(round(fpsReport,1))+ 'fps', (0,25), font,.75, (0,255,255, 2))
    cv2.imshow('Picture', frame)
    cv2.moveWindow('Picture', 0,0)
    output.write(frame)                 # video speichern

    if cv2.waitKey(1)== ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

##########################