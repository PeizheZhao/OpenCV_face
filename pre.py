import cv2

cap = cv2.VideoCapture(0)
#人脸检测_级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

while True:
    ret , frame = cap.read()
    if ret:
        #convert image(frame) to gray
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

        #draw a rectangle on face
        for(x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 2)

        cv2.imshow("Main window",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()