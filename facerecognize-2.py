import face_recognition as fr
import cv2
print(cv2.__version__)

donFace = fr.load_image_file('/home/nayon/Schreibtisch/facerecog/demoImages/known/Donald Trump.jpg')
donEncode = fr.face_encodings(donFace)[0] #Falls mehr als eine Encoding da ist, gibt er mehrere Arrays aus, deshalb wollen wir das erste Array einfach nehmen

nancyFace = fr.load_image_file('/home/nayon/Schreibtisch/facerecog/demoImages/known/Nancy Pelosi.jpg')
nancyEncode = fr.face_encodings(nancyFace)[0] #Falls mehr als eine Encoding da ist, gibt er mehrere Arrays aus, deshalb wollen wir das erste Array einfach nehmen

birxFace = fr.load_image_file('/home/nayon/Schreibtisch/facerecog/demoImages/known/Dr Birx.jpg')
birxEncode = fr.face_encodings(birxFace)[0] #Falls mehr als eine Encoding da ist, gibt er mehrere Arrays aus, deshalb wollen wir das erste Array einfach nehmen

mikeFace = fr.load_image_file('/home/nayon/Schreibtisch/facerecog/demoImages/known/Mike Pence.jpg')
mikeEncode = fr.face_encodings(mikeFace)[0] #Falls mehr als eine Encoding da ist, gibt er mehrere Arrays aus, deshalb wollen wir das erste Array einfach nehmen

paulFace = fr.load_image_file('/home/nayon/Schreibtisch/facerecog/demoImages/known/Paul McWhorter.jpg')
paulEncode = fr.face_encodings(paulFace)[0] #Falls mehr als eine Encoding da ist, gibt er mehrere Arrays aus, deshalb wollen wir das erste Array einfach nehmen

ronaldFace = fr.load_image_file('/home/nayon/Schreibtisch/facerecog/demoImages/known/Ronald Reagan.jpg')
ronaldEncode = fr.face_encodings(ronaldFace)[0] #Falls mehr als eine Encoding da ist, gibt er mehrere Arrays aus, deshalb wollen wir das erste Array einfach nehmen

seemaFace = fr.load_image_file('/home/nayon/Schreibtisch/facerecog/demoImages/known/Seema Verma.jpg')
seemaEncode = fr.face_encodings(seemaFace)[0] #Falls mehr als eine Encoding da ist, gibt er mehrere Arrays aus, deshalb wollen wir das erste Array einfach nehmen

surgeonFace = fr.load_image_file('/home/nayon/Schreibtisch/facerecog/demoImages/known/Surgeon General.jpg')
surgeonEncode = fr.face_encodings(surgeonFace)[0] #Falls mehr als eine Encoding da ist, gibt er mehrere Arrays aus, deshalb wollen wir das erste Array einfach nehmen


Encodings= [donEncode,nancyEncode, birxEncode, mikeEncode, paulEncode, ronaldEncode, seemaEncode, surgeonEncode]
Names= ['The Donald', 'Nancy Pelosi', 'Dr Birx', 'Mike Pence', 'Paul McWhorter', 'Ronald Reagan', 'Seema Verma', 'Surgeon General'] # Man kann auch das mit dem Filename machen, aber zum lernen erstmal so

font = cv2.FONT_HERSHEY_SIMPLEX
testImage = fr.load_image_file('/home/nayon/Schreibtisch/facerecog/demoImages/unknown/u9.jpg')
facePositions = fr.face_locations(testImage)
allEncodings = fr.face_encodings(testImage, facePositions) 
#Wir wollen jetzt für alle Gesichter Encodings haben, damit wir diese vergleichen und dann erkennen können
# Wir suchen die Facepositions in testImage und erstellen Encodings für jede Position. Das Array hat dann alle Gesichter Encoded

testImage = cv2.cvtColor(testImage, cv2.COLOR_RGB2BGR) # zu BGR weil OpenCV mit BGR arbeitet

for (top,right,bottom,left), face_encoding in zip(facePositions,allEncodings):# Weil man durch zwei Sätze an Variablen geht braucht man Zip
    name='unknown Person'
    matches = fr.compare_faces(Encodings, face_encoding) # Encodings sind unsere Trainieren Faces | Wir vergleichen Train Data und das Gesicht in unserem derzeitigem Loop
    if True in matches:
        first_match_index = matches.index(True) # Wenn man True in Matches findet, was ist der Index
        name=Names[first_match_index] #wenn don dann 0 wenn nancy dann 1
    cv2.rectangle(testImage,(left,top), (right, bottom), (0,0,255), 2)
    cv2.putText(testImage,name, (left,top-6), font, .75, (0,255,255), 1)

cv2.imshow('myWindow', testImage)
cv2.moveWindow('myWindow', 0, 0) # Window soll oben links erscheinen
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
