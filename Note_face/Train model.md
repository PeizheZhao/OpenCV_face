# Use camera to collect face
```python
import cv2
import time

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

count = 0

while True:
	ret , frame = cap.read()
	if ret:
		T1 = time.perf_counter()
		
		con_frame = cv2.rotate(frame , cv2.ROTATE_180)
		gray = cv2.cvtColor(con_frame , cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray , scaleFactor=1.5 , minNeighbors=5)
		for(x,y,w,h) in faces:
			cv2.rectangle(con_frame , (x,y) , (x+w,y+h) , (0,255,0) , 2)
			count+=1
			cv2.imwrite("./dataset/zpz/" + str(count) + ".jpg" , gray[y:y+h,x:x+w])
		
		T2 = time.perf_counter()
		fps = str(1000/((T2-T1)*1000))
		cv2.putText(con_frame , 'FPS:'+fps , (0,15) , cv2.FONT_HERSHEY_PLAIN , 1 , (0,255,0) ,1)	
		
		cv2.imshow("Main window" , con_frame)



	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	elif count == 50:
		break

cap.release()
cv2.destroyAllWindows()
```

# Use dataset to train model
```python
import os  
import cv2  
import numpy as np  
import pickle  
  
#the floder absolute path for this file  
#../OpenCV-face/  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
#concat dataset dir after BASE_DIR  
#../OpenCV-face/dataset/  
dataset_dir = os.path.join(BASE_DIR , "dataset")  
  
#count id  
current_id = 0  
#directory for name and id   {'zpz':3,...}  
label_ids = {}  
#store a list of roi numpy array of each image  
x_train = []  
#store a list of id of each image  
y_ids = []  
  
#create a face recoginzer  
recognizer = cv2.face.LBPHFaceRecognizer_create()  
#create a classifier  
classifiler = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")  
  
#root is a path dirs and files are lists  
#each time to loop, dirs and files creat lists of path which root shows  
for root , dirs , files in os.walk(dataset_dir):  
    for file in files:  
        #path for image  
        path = os.path.join(root , file)  
        #let cv2 read image  
        image = cv2.imread(path)  
        #conver image to gray  
        gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)  
        #convert images(gray) to numpy array  
        image_array = np.array(gray , "uint8")  
        #the last dir_name of root (username)  
        label = os.path.basename(root)  
  
        #if this is new label,create a new id for it  
        if not label in label_ids:  
            label_ids[label] = current_id  
            current_id += 1  
  
        #use classifiler to find roi and store roi in x_train,store id in y_ids  
        faces = classifiler.detectMultiScale(image_array , scaleFactor=1.5 , minNeighbors=5)  
        for(x,y,w,h) in faces:  
            roi = image_array[y:y+w,x:x+w]  
            x_train.append(roi)  
            y_ids.append(label_ids[label])  
  
#store labels_ids dirctory  
with open("./label.pickle","wb") as f:  
    pickle.dump(label_ids , f)  
  
#use x_train and y_ids to train model  
recognizer.train(x_train , np.array(y_ids))  
#save model  
recognizer.save("./mytrainer.xml")
```