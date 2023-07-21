import cv2
import pickle

cap = cv2.VideoCapture(0)
classifiler = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

#let recongnizer load training model
recognizer.read("./mytrainer.xml")

#prepare to exchange key and value
con_labels = {}
#open file and ecchange
with open('label.pickle','rb') as f:
    origin_labels = pickle.load(f)
    #exchange key and value
    con_labels = {v:k for k,v in origin_labels.items()}

while True:
    ret , frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        faces = classifiler.detectMultiScale(gray , scaleFactor=1.5 , minNeighbors=5)

        for (x,y,w,h) in faces:
            #create roi in gray image
            gray_roi = gray[y:y+h,x:x+w]
            #use recognizer to predict face in gray roi
            id_ , conf = recognizer.predict(gray_roi)
            if conf >= 50:
                cv2.rectangle(frame , (x,y) , (x+w,y+h) , (0,255,0) , 2)
                cv2.putText(frame , str(con_labels[id_]), (x,y-15) , cv2.FONT_HERSHEY_PLAIN , 3 , (0,255,0) ,2)

        cv2.imshow("Main window" , frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
