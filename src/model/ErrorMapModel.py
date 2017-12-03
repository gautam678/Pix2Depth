from GeneratorForErrorMap import GeneratorErrorMap
import os, os.path
import pprint
import glob
import random
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import numpy
import h5py
#"""
#from keras.applications.resnet50 import ResNet50
#from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input, decode_predictions
#"""
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,Input,concatenate,Maximum,Conv2D,Deconv2D, ZeroPadding2D, UpSampling2D
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import mean_absolute_error
import keras.backend as K
from math import pow
#STOP_LAYER=79
def customLoss(yTrue,yPred):
    if(yTrue.shape[1]!=None):
            return (K.mean(K.square((K.log(yTrue) - K.log(yPred))),axis=-1)-(0.5/pow(K.count_params(yTrue),2))*K.sum((K.log(yTrue) - K.log(yPred)),axis=-1))
    else:
            return K.sum(K.abs(yPred - yTrue), axis=-1)
def CreatErrorMapModel(input_shape, lastLayerActivation='hard_sigmoid', PercentageOfTrianable=70, bnAtTheend=True, lossFunction="mae"):
    STOP_LAYER=149
    print(STOP_LAYER)
    img_dim = (256,256,3)
    
    sharedResnet = applications.resnet50.ResNet50(include_top=False,input_shape=input_shape)
#    print(sharedResnet.summary())
#    print(sharedResnet.get_config())
#    print(sharedResnet.to_yaml())
#    for i,l in enumerate(sharedResnet.layers):
#    	print(i,"  ",l," ",l.output_shape,"\n")
#    
    BaseModel=Model(sharedResnet.input, sharedResnet.layers[STOP_LAYER].output)
    x = BaseModel.output
    print("x output shapr is ", x.shape)
    x=Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x=BatchNormalization()(x)
    x=Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x=BatchNormalization()(x)
    print("x output shapr is ", x.shape)    
    x = Deconv2D(8, (3, 3), strides=(2, 2), padding="same",  activation='relu')(x)
    x=BatchNormalization()(x)
    x = Deconv2D(4, (3, 3), strides=(2, 2), padding="same", activation='relu')(x)
    x=BatchNormalization()(x)
    x = Deconv2D(3, (3, 3), strides=(2, 2), padding="same", activation='relu')(x)
    x=BatchNormalization()(x)
    x = Deconv2D(3, (3, 3), strides=(2, 2), padding="same", activation='relu')(x)
    x=BatchNormalization()(x)
#    x = Deconv2D(3, (3, 3), strides=(2, 2), padding="same", activation='relu')(x)
#    x=BatchNormalization()(x)

    x = Deconv2D(3, (3, 3), strides=(2, 2), padding="same",  activation=lastLayerActivation)(x)
    if (bnAtTheend==True):
        x=BatchNormalization()(x)

    whole_model = Model(BaseModel.input, outputs=x)
#    print (resnet.layers)
    
    p=int((PercentageOfTrianable/100)*len(whole_model.layers))
    print(len(whole_model.layers), p)
    for layer in whole_model.layers[:p]:
    	layer.trainable = False
   
    for i,l in enumerate(whole_model.layers):
    	print(i,"  ",l," ",l.output_shape,"\n")

#    whole_model.summary() 
    print(lossFunction)
    if(lossFunction=="customLoss"):
    	lossFunction=customLoss
    whole_model.compile(loss=lossFunction,optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['mae', 'acc'])
    return  whole_model

