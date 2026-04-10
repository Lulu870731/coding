import cv2
from cv2 import face
import numpy as np
from deepface import DeepFace

#載入Haar人臉檢測分類器
oface=cv2.CascadeClassifier('face.xml')
glasses=cv2.CascadeClassifier('glasses.xml')
cap=cv2.VideoCapture(0)

img=cv2.imread('me.jpg')#把自己的照片取名為me.jpg，並引入
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=oface.detectMultiScale(gray,1.3,5)

if len(faces)==0:
    print("未偵測到參考人臉")
    exit()
(x, y, w, h) = faces[0]
aface=gray[y:y+h, x:x+w]
# 訓練 LBPHFaceRecognizer
recognizer=face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.train([aface], np.array([0]))  # 標記為0


while True:
    #從攝影機讀取影像
    ret,frame=cap.read()
    if not ret:
        print("無法從攝影機讀取影像。")
        break
    #將影像轉為灰階
    gray1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face1=oface.detectMultiScale(gray1,scaleFactor=1.08,minNeighbors=5,minSize=(32,32)) #(載入gray,縮小的尺寸,相鄰的框框數,定義像數)
    try
        for(x,y,w,h) in face1:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            #擷取人臉區域
            gray2=gray1[y:y+h,x:x+w]
            #使用辨識器進行人臉比對
            label,confidence=recognizer.predict(gray)
            #根據信心值判斷是否為同一個人
            if confidence<90: #設定信心值門檻
                text="Match: Me"
                color=(0,255,0) #綠色
            else:
                text="Not Me"
                color = (0,0,255) #紅色

            #在影像上顯示結果
            cv2.putText(frame,text,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            #偵測人臉區域中的眼睛
            eyes_with_glasses=glasses.detectMultiScale(gray2,scaleFactor=1.1,minNeighbors=5,minSize=(20,20))
            for (ex,ey,ew,eh) in eyes_with_glasses:
                cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey + eh),(255, 0, 0), 2)
                #在框框旁顯示"glasses"
               cv2.putText(frame, "glasses", (x+ex,y+ey-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
            # 一次性進行所有分析，減少重複呼叫
            analysis=DeepFace.analyze(img,actions=['emotion', 'age', 'race', 'gender'])

            # 提取分析結果
            emotion=analysis['emotion']['dominant_emotion']
            age=analysis['age']
            race=analysis['race']['dominant_race']
            gender=analysis['gender']

            # 顯示結果
            print(f"Emotion:{emotion}")
            print(f"Age:{age}")
            print(f"Race:{race}")
            print(f"Gender:{gender}")

    #顯示影像
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #關閉視窗
cap.release()
cv2.destroyAllWindows()
