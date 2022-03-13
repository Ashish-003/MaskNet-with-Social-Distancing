from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import time
import cv2
import os
import math

BASE_PATH = "./"

def detect_and_predict_mask(frame, faceNet, maskNet, conf_threshold):
    '''
    Detects mask on face detections greater than conf_threshold
    :return: tuple of locations and probabilities of detections 
    '''
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # take high confidence detections
        if confidence > conf_threshold:
            # bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # get face ROI and preprocess it 
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
        
    if faces:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def get_output_names(net)->list:
    '''
    Get the names of the output layers
    '''
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

if __name__ == "__main__":
    scale_percent = 20 # percent of original size
    width = 0
    height = 0
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    rg_dict = { "RED": (0, 0, 255), "GREEN": (0, 255, 0)}
    
    labels_path = BASE_PATH + "Models/coco.names"
    weights_path = BASE_PATH + "Models/yolov3.weights"
    config_path = BASE_PATH + "Models/yolov3.cfg"

    with open(labels_path, 'rt') as f:
        LABELS = f.read().rstrip('\n').split('\n')
    
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        exit()
    else: #get dimension info
        width =  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dim = (width, height)
        print('Original Dimensions : ',dim)
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        dim = (width, height)
        print('Resized Dimensions : ', dim)

    confidence=0.4

    print("[INFO] loading face detector model...")
    cnfg_path = './Models/deploy.prototxt'
    weightsPath = './Models/res10_300x300_ssd_iter_140000.caffemodel'
    faceNet = cv2.dnn.readNet(cnfg_path, weightsPath)
    
    print("[INFO] loading face mask detector model...")
    model_store_dir=BASE_PATH + "Models/mask_detector.model"
    maskNet = load_model(model_store_dir)

    print("[INFO] starting video stream...")
    # vid_stream = VideoStream(src=0).start()
    vid_stream = cv2.VideoCapture(0)
    time.sleep(2.0)
   
    # loop over the frames from the video stream
    iter=0
    while cv2.waitKey(1) < 0:

        (hasframe, frame) = vid_stream.read()
        if not hasframe:
            cv2.waitKey(3000)
            cap.release()
            break

        frame = imutils.resize(frame, width=1200)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        (H, W) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (224, 224), [0,0,0], swapRB=True, crop=False)
        net.setInput(blob)
        # start = time.time()
        layerOutputs = net.forward(get_output_names(net))
        # end = time.time()
        # print("Frame Prediction Time : {:.6f} seconds".format(end - start))
        
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > 0.4 and classID == 0:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        if iter % 3 == 0:
            # Non Max Suppression to eliminate overlapping boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
            ind = []
            a = []
            b = []
            for i in range(len(classIDs)):
                if classIDs[i] == 0:
                    ind.append(i)
            if idxs:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    a.append(x)
                    b.append(y)
            
            # Check Social Distancing
            distance = []
            nsd = []
            for i in range(len(a) - 1):
                for k in range(1, len(a)):
                    if k == i:
                        break
                    else:
                        x_dist = (a[k] - a[i])
                        y_dist = (b[k] - b[i])
                        d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                        distance.append(d)
                        if d <= 6912:         
                            nsd.append(i)
                            nsd.append(k)
                        nsd = list(dict.fromkeys(nsd))
            
            # Mark distancing detections
            for i in nsd:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), rg_dict["RED"], 2)
                cv2.putText(frame, "UNSAFE", (x, y - 5), FONT, 0.5, rg_dict["RED"], 2)

            if idxs:
                for i in idxs.flatten():
                    if i not in nsd:
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), rg_dict["GREEN"], 2)
                        cv2.putText(frame, "SAFE", (x, y - 5), FONT, 0.5, rg_dict["GREEN"], 2)
                        
        cv2.putText(frame, "Social Distance Violaters: {}".format(len(nsd)), (10, frame.shape[0] - 25),
                    FONT, 0.85, rg_dict["RED"], 2)
        cv2.putText(frame, "MASK DETECTION WITH SOCIAL DISTANCING", (200, 45), FONT, 1, rg_dict["GREEN"], 2)        
        
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, 0.5)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            # print(box,pred)
            if len(pred) == 2:
                mask = pred[0]
                withoutMask = pred[1]
            elif len(pred) == 1:
                mask = pred[0]
                withoutMask = 0
            label = "MASK WORN" if mask > withoutMask else "MASK NOT WORN"
            color = rg_dict["GREEN"] if label == "MASK WORN" else rg_dict["RED"]
            
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            
            # display the label and bounding box
            cv2.putText(frame, label, (startX, startY - 10), FONT, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # cv2.namedWindow('DETECTOR', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('DETECTOR', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('DETECTOR', frame)
        key = cv2.waitKey(1) & 0xFF
        # Exit using the q key
        if key == ord("q"):
            break
    vid_stream.release()
    cv2.destroyAllWindows()