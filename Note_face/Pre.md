# Shell 基础
```
//setting SSH VNC
sudo raspi-config 

//creat python virture environment 
python -m venv ./face (PATH)
//activate encironment
source ./bin/activate

//use pip install pyhton packets
sudo apt install libatlas-base-dev
sudo apt install libjasper-dev
sudo apt install libhdf5-dev

pip install opencv-contrib-python==4.5.4.60

//detect carmera
ls /dev/video*

//count time (FPS)
T1 = time.perf_counter()
```


# 简单框选出人脸

```python
import cv2  
  
cap = cv2.VideoCapture(0)  
#人脸检测_级联分类器  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")  
  
while True:  
    ret , frame = cap.read()  
    if ret:  
        #convert image(frame) to gray  
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
        #检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5) 
        #draw a rectangle on face  
        for(x,y,w,h) in faces:  
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 2)  
  
        cv2.imshow("frame",frame)  
  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
  
cap.release()  
cv2.destroyAllWindows()
```

***
## cv2.CascadeClassifier

eg:
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_alt2.xml")  
```

路径:`cv2/data/xxx.xml`

CascadeClassifier，是Opencv中做人脸检测的时候的一个级联分类器。并且既可以使用[[Theory#Haar特征]]，也可以使用LBP特征。


***
## face_cascade.detectMultiScale

eg:
```python
faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
```

cvHaarDetectObjects是opencv1中的函数，opencv2中人脸检测使用的是 detectMultiScale函数。它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示），函数由分类器对象调用

参数1：image--待检测图片，一般为灰度图像加快检测速度；
参数2：objects--被检测物体的矩形框向量组；
参数3：scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
参数4：minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
这种设定值一般用在用户自定义对检测结果的组合程序上；
参数5：flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为
CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，因此这些区域通常不会是人脸所在区域；
参数6、7：minSize和maxSize用来限制得到的目标区域的范围。

***

