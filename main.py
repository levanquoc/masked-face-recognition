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
data_nomask=pickle.loads(open("encodings_nomask.pickle","rb").read())
data = pickle.loads(open("encodings.pickle","rb").read())
landmark_detect=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def convnet_model_():
    vgg_model = applications.VGG16(weights=None, include_top=False, input_shape=(96, 96, 3))
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda x_: K.l2_normalize(x,axis=1))(x)
#     x = Lambda(K.l2_normalize)(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model

def deep_rank_model():
    convnet_model = convnet_model_()

    first_input = Input(shape=(96, 96, 3))
    first_conv = Conv2D(96, kernel_size=(8,8), strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(first_conv)
    first_max = Flatten()(first_max)
#     first_max = Lambda(K.l2_normalize)(first_max)
    first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)

    second_input = Input(shape=(96, 96, 3))
    second_conv = Conv2D(96, kernel_size=(8,8), strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7), strides=(4,4), padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)
#     second_max = Lambda(K.l2_normalize)(second_max)
                       
    merge_one = concatenate([first_max, second_max])
    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    emb = Dense(100)(emb)
    l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)
#     l2_norm_final = Lambda(K.l2_normalize)(emb)
                        
    final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model

model=deep_rank_model()
model.load_weights('triplet_weight.hdf5')
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
                cv2.imwrite('cuong.png',markface)
                markface=cv2.resize(markface,(96,96))
                #markface = cv2.cvtColor(markface, cv2.COLOR_BGR2RGB)
                markface=markface/255
                markface=np.expand_dims(markface,axis=0)
                emb100=model.predict([markface,markface,markface])
                dist=[]
                for i in range(0,len(data['encodings'])):
                    dist.append(np.linalg.norm(emb100-data['encodings'][i]))
                index=dist.index(min(dist))    
                print(dist,index)
                cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255), 2)
                cv2.putText(img,data['names'][index], (x, y - 5), font, 3,(0,255,0), 1)
            else:
                emb128=face_recognition.face_encodings(face,[(0,boxes[i][2],boxes[i][3],0)])
                matches=face_recognition.compare_faces(data_nomask["encodings"],emb128[0],tolerance=0.5)
                #print(matches)
                counts={}
                name="unknown"
                id="unknown"
                if True in matches:
                    matchesID=[i for (i,b) in enumerate(matches,start=0)if b]
                    for i in matchesID:
                        id=data_nomask['ids'][i]
                        counts[id]=counts.get(id,0)+1
                    id=max(counts,key=counts.get)
                for i,idname in enumerate(data_nomask["ids"]):
                    if idname==id:
                        name=data_nomask["names"][i]
                cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(255,0,0),2)          
                cv2.putText(img,name,(int(x),int(y-40)), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1,cv2.FILLED)         
    cv2.imshow("Image", img)  
    
    #print(face)
def image_detect(img_path):
    model,classes,colors,output_layers=load_yolo()
    image,height,width,chanels=load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    
    boxes,confs,class_ids=get_box_dimensions(outputs,height,width)
   
    
    draw_labels(boxes,confs,colors,class_ids,classes,image)
    
    while True:
        key=cv2.waitKey(1)
        if key==27:
            break
def compute_dist(a,b):
    return np.sum(np.square(a-b))
def webcam_detect():

    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(0)
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
    
    