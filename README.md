# MaskNet with Social Distancing

**Project to detect face masks with social distancing in real time using YOLOv3, Haar Cascade and OpenCV**
## Mask detection using Haar Cascade
1. Build and train CNN model with data augmentation to improve performance
2. Capture the faces using Haar Cascade using the frames from input stream
3. Use the trained model to detect mask on the faces

**Instructions:**

Build and train the CNN model:

`` python train_model.py `` 

Use trained model for detection:

`` python detector.py ``

## Mask detection with social distancing using YOLO
1. Capture the frames from video using OpenCV and pass the frame through YOLOv3 to get the persons in the frame
2. Social distancing is calculated using Euclidean distance between the persons detected
3. The person detections are then passed through face_detection.model to extract the faces
4. The pre-trained mask_detector.model uses the extracted faces to detect mask

**Instructions:**

Run the program:

`` python main.py ``

Exit program by pressing 'q' key
