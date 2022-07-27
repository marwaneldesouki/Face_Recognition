import encodings
import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import  datetime
from PIL import ImageGrab
import threading
import concurrent.futures

def resize(img, size) :
    try:
        width = int(img.shape[1]*size)
        height = int(img.shape[0] * size)
        dimension = (width, height)
        return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)
    except:
        pass

path = 'images'
personsImg = []
personsName = []
persons_num= 0
photos_num = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")
for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()  #the name of dir(person name)
            curimg = cv2.imread(f'{path}')
            persons_num +=1
            personsImg.append(curimg) 
            personsName.append(label)
            photos_num+=1
print(persons_num)
# print(personsImg)

def findEncoding(images) :
    imgEncodings = []
    index = 0
    for img in images :
        try:
            img= resize(img, 0.50)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodeimg = face_rec.face_encodings(img)[0]
            imgEncodings.append(encodeimg)
        except Exception as ex:
            print("error in image:"+str(index))
        index+=1
    return imgEncodings

EncodeList = findEncoding(personsImg)
# vid = cv2.VideoCapture(0) #-uncomment to enable video capture
while True :
    # success, frame =vid.read() #-uncomment to enable video capture
    frame = np.array(ImageGrab.grab(bbox=(0,40,1500,1500))) #comment to disable screen capture
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Smaller_frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    facesInFrame = face_rec.face_locations(Smaller_frames)
    encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame)
    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
            matches = face_rec.compare_faces(EncodeList, encodeFace)
            facedis = face_rec.face_distance(EncodeList, encodeFace)
            matchIndex = np.argmin(facedis)
            # print(matches,facedis,matchIndex)
            try:
              if matches[matchIndex] :
                    name = personsName[matchIndex].upper()
                    if name == "DESO":
                        name = "Legend Deso"
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
              else:
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.rectangle(frame, (x1, y2-25), (x2, y2), (255, 0, 0), cv2.FILLED)
                cv2.putText(frame,"unknown", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            except:
                y1, x2, y2, x1 = faceloc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0 , 0), 3)
                cv2.rectangle(frame, (x1, y2-25), (x2, y2), (255, 0, 0), cv2.FILLED)
                cv2.putText(frame,"unknown", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('video',frame)
    if cv2.waitKey(20) & 0xFF ==ord('q') :
        break
vid.release()
cv2.destroyAllWindows()

# print(personsName)
