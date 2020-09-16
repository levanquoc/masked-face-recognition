#  MASKED FACE RECOGNITION
# Deep Learning Based Masked Face Recognition Using Triplet Loss 
## Getting started

Triplet loss is a loss function for machine learning algorithms where a baseline (anchor) input is compare 
to a positive (truthy) input and a negative (falsy) input. The distance from the baseline (anchor) input to 
the positive (truthy) input is minimized, and the distance from the baseline (anchor) input to 
the negative (falsy) input is maximized.For more details, you can refer to this [paper](https://arxiv.org/pdf/1503.03832.pdf)

## Architecture
![alt](https://i.imgur.com/RaMpNCm.png)
## OpenCV Deep Neural Networks (dnn module)
OpenCV dnn module supports running inference on pre-trained deep learning models from popular frameworks such as TensorFlow, Torch, Darknet and Caffe.
## Prerequisites
* Opnecv
* Tensorflow
* Keras
* Numpy
* Matplotlib
* Face_recognition
## Usage
* Clone this repository
```bash
$ git clone https://github.com/levanquoc/masked-face-recognition.git
```
* Next,you can get data from webcam by:
```bash
$ cd getdata
$ python3 getdata_from_webcam.py
```

* Next,you save data from getdata folder to dataset folder 
* You run encodings.py to perform face encodings
```bash
$ cd ..
$ python3 encodings.py
```
* Finally,you run main.py to check result
``` bash
$ python3 main.py
```
## Result

![Alt](images/CAPTURE.PNG)
