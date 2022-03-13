import cv2
import numpy as np
from keras.models import load_model
model=load_model("./model-010.model")
results={0:'without mask',1:'mask'}
rg_dict={0:(0,0,255),1:(0,255,0)}
sz = 4
cap = cv2.VideoCapture(0) 
haarcascade = cv2.CascadeClassifier('./Models/haarcascade_frontalface_default.xml')
while True:
    (hasframe, frame) = cap.read()
    frame=cv2.flip(frame,1,1) 
    if not hasframe:
        cap.release()
        break
    img_resz = cv2.resize(frame, (frame.shape[1]//sz, frame.shape[0]//sz))
    faces = haarcascade.detectMultiScale(img_resz)
    for f in faces:
        (x, y, w, h) = [v * sz for v in f] 
        
        face_img = frame[y:y+h, x:x+w]
        fc_resz=cv2.resize(face_img,(150,150))
        face_in = np.vstack([np.reshape(fc_resz/255.0,(1,150,150,3))])
        result=model.predict(face_in)
        
        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(frame,(x,y),(x+w,y+h),rg_dict[label],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),rg_dict[label],-1)
        cv2.putText(frame, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow('DETECTOR', frame)
    key = cv2.waitKey(1) & 0xFF
    # Exit program using 'q'
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()