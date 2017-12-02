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
import matplotlib.pyplot as plt
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
from ErrorMapModel import CreatErrorMapModel
import keras
import argparse
#STOP_LAYER=79
STOP_LAYER=149
print(STOP_LAYER)
img_dim = (256,256,3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Error Train model')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--DataSetPath', type=str, default="../../data/nyu_depth_v2_labeled.mat", help="path to dataset")
    parser.add_argument('--lastLayerActivation', type=str, default='hard_sigmoid', help="Activation of the lastLayer")
    parser.add_argument('--PercentageOfTrianable', type=int, default=70, help="Percentage of Triantable Layers")
    parser.add_argument('--SpecificPathStr', type=str, default='Org', help="PathStr")
    args = parser.parse_args()
    path = args.DataSetPath


    logpath=os.path.join('../../log','ErrorMapWith'+args.lastLayerActivation+str(args.PercentageOfTrianable)+'UnTr'+args.SpecificPathStr)
    modelPath=os.path.join('../../models','ErrorMapwith'+args.lastLayerActivation+str(args.PercentageOfTrianable)+'Untr'+args.SpecificPathStr)
    print(logpath)
    print(modelPath)    
    os.makedirs(logpath, exist_ok=True)
    os.makedirs(modelPath, exist_ok=True)
    f = h5py.File(path)
    ErrorMapGen , length =GeneratorErrorMap(img_dim, batch_size=1,modelEpoch=130, threshold=0.1,f=f)
    x1, ErrorMap = ErrorMapGen.__next__() 
    print(ErrorMap.shape)
#    img, ErrorMap = ErrorMapGen.__next__()                             # Python 3 syntax. Use .next() instead for Python 2.
####################################################
#    nb_train_samples = 20
    nb_train_samples = 2000
    nb_validation_samples = 800
#    nb_validation_samples = 8
    epochs = 20
#    epochs = 1
    batchSize=args.batch_size

    indexes=[i  for i in range(1448)]
    np.random.shuffle(indexes)
    indexesTrain=indexes[1:int(0.8*len(f))]
    indexesTest=indexes[int(0.8*len(f)):]
    (batchTrainGen, length)=GeneratorErrorMap(img_dim, batch_size=batchSize,modelEpoch=130, threshold=0.1,f=f,indexes=indexesTrain)
    (batchValidGen, length) =GeneratorErrorMap(img_dim, batch_size=batchSize,modelEpoch=130, threshold=0.1,f=f,indexes=indexesTest)

    for i in range(10):
    	x1, ErrorMap = batchTrainGen.__next__()
    	x2, ErrorMap2 = batchValidGen.__next__()
#    	print(x1)
#   	print(ErrorMap)
#    	print(x1.shape)
#    	print(ErrorMap.shape)
#    	print(len(batchTrainGen.__next__()))
#    	input("Press Enter to continue...")
    whole_model=CreatErrorMapModel(input_shape=x1.shape[1:],lastLayerActivation=args.lastLayerActivation, PercentageOfTrianable=args.PercentageOfTrianable)
    history=whole_model.fit_generator(batchTrainGen, samples_per_epoch=nb_train_samples,epochs=epochs,validation_data=batchValidGen,nb_val_samples=nb_validation_samples,       callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join(modelPath,'ErrorMap_weightsBestLoss.h5'), monitor='val_loss', verbose=1, save_best_only=True),
            keras.callbacks.ModelCheckpoint(os.path.join(modelPath,'ErrorMap_weightsBestAcc.h5'), monitor='acc', verbose=1, save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
            keras.callbacks.TensorBoard(log_dir=logpath, histogram_freq=0, batch_size=batchSize, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        ],)
    ErrorMap_weights_path = os.path.join(modelPath,'ErrorMap_weights.h5' )
    whole_model.save_weights(ErrorMap_weights_path, overwrite=True)
    plt.plot(history.history['loss'])
    PlotPath = os.path.join(logpath,'LossPlot.pdf' )
    plt.savefig(PlotPath,bbox_inches='tight')

