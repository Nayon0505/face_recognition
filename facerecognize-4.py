import pickle
import cv2
import face_recognition as fr
import os
print(cv2.__version__)

Encodings= []  # Das ist unser Array für die Encodings, Es ist ein Array aus Arrays, da Encodings auch Arrays sind
Names= [] # Wir brauchen die Namen für die Encodings
j= 0 #Counter

image_dir= '/home/nayon/Schreibtisch/facerecog/demoImages/known' # Die Directory unseres Known Verzeichnisses, Jetzt wollen wir da einmal durchgehen alle Bilder encoden, die Namen herausholen und die dann zusammenzuführen

for root, dirs, files in os.walk(image_dir): #mit der os Library und walk kann ich durch die Directories durchgehen | Wir laufen hier nur durch files
    print(files)
    for file in files:
        path=os.path.join(root,file)
        print(path)
        name= os.path.splitext(file)[0] # Teilt den Filenamen und nimmt die erste Postition (Nayon.JPG) der Punkt trennt die Positionen
        print(name)
        person = fr.load_image_file(path)
        encoding = fr.face_encodings(person)[0] # Gibt ein Array zurück, weil er nach mehrere Gesichter sucht und wir nehmen das erste sonst wird es ein Array in einem Array oderso
        Encodings.append(encoding)
        Names.append(name)
print(Names)

#pickle! Macht keinen Stress wegen Formaten
with open('train.pkl', 'wb') as f: # wb steht für write bytes, f ist as Objekt auf das wir zugreifen, wenn wir unser Pickle readen oder writen wollen
    pickle.dump(Names,f)
    pickle.dump(Encodings,f)
Encodings=[] # Wir müssen Names und Encodings löschen, damit nicht die alten Daten benutzt werden
Names=[]
with open('train.pkl', 'rb') as f:
    Names= pickle.load(f)
    Encodings= pickle.load(f)
    

# Jetzt sind alle aus dem Folder 'known' für das Modell trainiert

font = cv2.FONT_HERSHEY_SIMPLEX

image_dir = '/home/nayon/Schreibtisch/facerecog/demoImages/unknown'
for root, dirs, files in os.walk(image_dir):
    for file in files:
        print(root)
        print(file)
        testImagePath=os.path.join(root,file)
        testImage= fr.load_image_file(testImagePath)
        facePositions = fr.face_locations(testImage) #Es returned ein Array aller Gesichter auf dem Bild
        allEncodings = fr.face_encodings(testImage, facePositions) # Wir haben jetzt die Positions und encoden die Positions auf dem Image
        testImage = cv2.cvtColor(testImage, cv2.COLOR_RGB2BGR) #Konvertieren zu BGR wegen OpenCV

        for (top,right,bottom,left), face_encoding in zip(facePositions, allEncodings):   # Bei jedem Loop gehen wir durch die Gesichter und für jedes Gesicht haben wir das Encoding
            name= 'unkown Person'
            matches= fr.compare_faces(Encodings,face_encoding) # Wir vergleichen alle unsere bekannten Gesichter Encodings und vergleichen es mit dem derzeitigem Encoding
            if True in matches: # Die compare_faces Methode markiert alle erkannten Gesichter mit True also,: wenn True in Matches ist dann
                first_match_index = matches.index(True)
                name= Names[first_match_index]
            cv2.rectangle(testImage,(left,top), (right, bottom), (0,0,255), 2) # OpenCV und Facerecognition ordnen diese (right,left,bottom usw.) verschieden. Darauf muss geachtet werden
            cv2.putText(testImage, name, (left, top-6), font, .75, (0,255,255), 2)

        cv2.imshow('Picture',testImage)
        cv2.moveWindow('Picture',0,0)
        if cv2.waitKey(0) == ord('q'):
          cv2.destroyAllWindows()


