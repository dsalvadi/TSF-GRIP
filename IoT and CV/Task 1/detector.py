# Image Object Detector

import cv2
import numpy as np

# Threshold to detect objects
thres = 0.5

img = cv2.imread('basketball_game.jpg')

class_names = []
class_file = 'coco.names'
with open(class_file, "rt") as f:
    class_names = f.read().rsplit('\n')
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_file = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights_file, config_file)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

class_IDs, confs, bbox = net.detect(img, confThreshold = thres)
print(class_IDs, bbox)

for class_ID, confidence, box in zip(class_IDs.flatten(), confs.flatten(), bbox):
    cv2.rectangle(img, box, color = colors[class_ID-1], thickness = 2)
    cv2.putText(img, class_names[class_ID - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, colors[class_ID-1], 2)
    cv2.putText(img, str(round(confidence*100, 2)) + '%', (box[0] + 150, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, colors[class_ID-1], 2)

cv2.imshow("Output", img)
cv2.waitKey(0)