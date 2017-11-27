from GeneratorForErrorMap import GeneratorErrorMap
import os, os.path
import pprint
import glob
import random
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import numpy
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
STOP_LAYER=79
print(STOP_LAYER)
img_dim = (256,256,3)

if __name__ == '__main__':
    ErrorMapGen , length =GeneratorErrorMap(img_dim, batch_size=1,modelEpoch=130, threshold=0.1)
    x1, ErrorMap = ErrorMapGen.__next__() 
    print(ErrorMap.shape)
#    img, ErrorMap = ErrorMapGen.__next__()                             # Python 3 syntax. Use .next() instead for Python 2.
####################################################
    sharedResnet = applications.resnet50.ResNet50(include_top=False,input_shape=x1.shape[1:])
#    print(sharedResnet.summary())
#    print(sharedResnet.get_config())
#    print(sharedResnet.to_yaml())
#    for i,l in enumerate(sharedResnet.layers):
#    	print(i,"  ",l," ",l.output_shape,"\n")
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
    x = Deconv2D(3, (3, 3), strides=(2, 2), padding="same",  activation='hard_sigmoid')(x)
    x=BatchNormalization()(x)

    whole_model = Model(BaseModel.input, outputs=x)
#    print (resnet.layers)
    p=int(0.7*len(whole_model.layers))
    print(len(whole_model.layers), p)
    for layer in whole_model.layers[:p]:
    	layer.trainable = False
   
    for i,l in enumerate(whole_model.layers):
    	print(i,"  ",l," ",l.output_shape,"\n")
    nb_train_samples = 2000
    nb_validation_samples = 800
    epochs = 50
    batchSize=1
    whole_model.compile(loss='mean_absolute_error',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
    
 
    (batchTrainGen, length)=GeneratorErrorMap(img_dim, batch_size=batchSize,modelEpoch=130, threshold=0.1)
    (batchValidGen, length) =GeneratorErrorMap(img_dim, batch_size=batchSize,modelEpoch=130, threshold=0.1)
#    batchValidGen=batchTrainGen
    for i in range(10):
    	x1, ErrorMap = batchTrainGen.__next__()
    	x2, ErrorMap2 = batchValidGen.__next__()
    	print(x1)
    	print(ErrorMap)
    	print(x1.shape)
    	print(ErrorMap.shape)
    	print(len(batchTrainGen.__next__()))
#    	input("Press Enter to continue...")
    whole_model.fit_generator(batchTrainGen, samples_per_epoch=nb_train_samples,epochs=epochs,validation_data=batchValidGen,nb_val_samples=nb_validation_samples)
