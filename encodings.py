from imutils import paths
import cv2
import os
import pickle
from keras.models import Sequential,Model 
from keras.layers import *
from keras.optimizers import *
from keras import applications
from keras import backend as K 
import tensorflow


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
imagePaths = list(paths.list_images("Dataset"))
knownEncodings = []
knownNames = []
knownIDs = []
for (i, imagePath) in enumerate(imagePaths):
    info  = imagePath.split(os.path.sep)[-2]
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    info = (info.split('_'))
    name = info[0]
    id = info[1]
    print(info)
    face = cv2.imread(imagePath)
    face=cv2.resize(face,(96,96))
    face=face/255
    
    face=np.expand_dims(face,axis=0)
    encodings = model.predict([face,face,face])
    if(len(encodings) != 1):
        print("error image")
        continue
    for encoding in encodings:
        print("Recognition Ok !")
        knownEncodings.append(encoding)
        knownNames.append(name)
        knownIDs.append(id)
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames, "ids": knownIDs}
print(data)
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()