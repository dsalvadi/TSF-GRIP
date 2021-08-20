# Real-Time Object Detector

import cv2
import numpy as np

# Threshold to detect objects
thres = 0.5
nms_threshold = 0.2 

cap = cv2.VideoCapture(0)   # Setting video capture device
cap.set(3, 1280)            # width
cap.set(4, 720)             # height
cap.set(10, 150)            # brightness

# Creating a list of classification names
class_names = []
class_file = 'coco.names'
with open(class_file, "rt") as f:
    class_names = f.read().rsplit('\n')

# creating a list of colors based on the classifications
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Configuring file paths to model files
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_file = 'frozen_inference_graph.pb'


# Configuring Detection Model and setting parameters
net = cv2.dnn_DetectionModel(weights_file, config_file)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)    # OpenCV reads images in BGR

while True:
    success, img = cap.read()
    class_IDs, confs, bbox = net.detect(img, confThreshold = thres)
    # print(class_IDs, bbox)
    # converting numpy arrays to lists
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float, confs))
    
    # using non max suppression (nms) to eliminate overlaps
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # add bounding box to image
        cv2.rectangle(img, (x, y), (x + w, h + y), color = colors[i], thickness = 2)
        # add classification label
        cv2.putText(img, class_names[class_IDs[i][0] - 1], (box[0] + 10, box[1] + 30), \
                cv2.FONT_HERSHEY_COMPLEX, 1, colors[i], 2)
        # add confidence percentage
        cv2.putText(img, str(round(confs[i]*100, 2)) + '%', (box[0] + 200, box[1] + 30), \
                cv2.FONT_HERSHEY_COMPLEX, 1, colors[i], 2)


    # if len(class_IDs) != 0:
    #     for class_ID, confidence, box in zip(class_IDs.flatten(), confs.flatten(), bbox):
    #         # add bounding box to image
    #         cv2.rectangle(img, box, color = (0, 255, 0), thickness = 2)
    #         # add classification label
    #         cv2.putText(img, class_names[class_ID - 1], (box[0] + 10, box[1] + 30), \
    #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    #         # add confidence percentage
    #         cv2.putText(img, str(round(confidence*100, 2)) + '%', (box[0] + 150, box[1] + 30), \
    #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)