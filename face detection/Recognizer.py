
import cv2
import os
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
recognizer = cv2.face.LBPHFaceRecognizer_create()
assure_path_exists("trainer/")
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,0,255), 4)
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if(Id == 1):
         Id = "Arvind {0:.2f}%".format(round(100 - confidence, 2))
        cv2.rectangle(img, (x-22,y-90), (x+w+22, y-22), (0,0,255), 2)
        cv2.putText(img, str("Arvind"), (x,y+40), font, 1, (255,255,255), 3)

    cv2.imshow('img',img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
