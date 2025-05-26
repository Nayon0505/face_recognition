import pickle
import cv2
import face_recognition as fr
import os
print(cv2.__version__)

Encodings= []  # Das ist unser Array für die Encodings, Es ist ein Array aus Arrays, da Encodings auch Arrays sind
Names= [] # Wir brauchen die Namen für die Encodings

image_dir= '/home/nayon/Schreibtisch/facerecognition/demoImages/known' # Die Directory unseres Known Verzeichnisses, Jetzt wollen wir da einmal durchgehen alle Bilder encoden, die Namen herausholen und die dann zusammenzuführen

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