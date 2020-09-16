import face_recognition
import cv2
import numpy as np
import argparse
import time
import dlib
from keras.models import Sequential,Model 
from keras.layers import *
from keras.optimizers import *
from keras import applications
from keras import backend as K 
from imutils import face_utils
import pickle
import numpy as np

landmark_detect=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def load_yolo():
    net=cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
    classes=[]
    with open ("yolo.names","r") as f:
        classes=[line.strip() for line in f.readlines()]
    layers_names=net.getLayerNames()
    output_layers=[layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors=np.random.uniform(0,255,size=(len(classes),3))
    return net,classes,colors,output_layers
  
def load_image(img_path):
    img=cv2.imread(img_path)
    
    img=cv2.resize(img,None,fx=0.4,fy=0.4)
    height,width,chanels=img.shape
    return img,height,width,chanels
    
def detect_objects(img,net,output_layers):
    blob=cv2.dnn.blobFromImage(img,scalefactor=0.000392,size=(320,320),mean=(0,0,0),swapRB=True,crop=False)
    net.setInput(blob)
    outputs=net.forward(output_layers)
    return blob,outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            #print(scores)
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids
    
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    global count
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            face=img[y:y+h, x:x+w] 
            
            if label=='face_mask':
                rect=dlib.rectangle(int(0),int(0),int(0+w),int(0+h))
                landmark=landmark_detect(face,rect)

                landmark = face_utils.shape_to_np(landmark)
               
                #markface=face[(landmark[18][1])-20:(landmark[29][1]),(landmark[18][0])-10:(landmark[26][0])]
                markface=face[int(landmark[18][1]):int(landmark[29][1]),int(landmark[18][0]):int(landmark[26][0])]
                cv2.imwrite(str(count)+'.png',markface)
                count=count+1
    cv2.imshow("Image", img) 
    #print(face)

def webcam_detect():

    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
    global count
    count=0
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

        
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()     
    
def main():
    #img_path="nam.jpg"
    #image_detect(img_path)
    webcam_detect()
    
    
    #start_video("quoc.mp4")
if __name__=='__main__':
    main()
    
    