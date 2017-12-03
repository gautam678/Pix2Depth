import os
import sys
import time
import numpy as np
import models
import keras
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
# Utils
sys.path.append("../utils")
import general_utils
import data_utils

from ErrorMapModel import CreatErrorMapModel
import shutil

def trainDepthMap(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    nb_train_samples = kwargs["nb_train_samples"]
    nb_validation_samples = kwargs["nb_validation_samples"]
    epochs = kwargs["nb_epoch"]
    model_name = kwargs["model_name"]
    lastLayerActivation=kwargs["lastLayerActivation"]
    PercentageOfTrianable=kwargs["PercentageOfTrianable"]
    SpecificPathStr=kwargs["SpecificPathStr"]
    lossFunction=kwargs["lossFunction"]
    if(kwargs["bnAtTheend"]!="True"):
         bnAtTheend=False
    else:
         bnAtTheend=True
    # Setup environment (logging directory etc)
    #general_utils.setup_logging(model_name)

    # Load and rescale data
    #X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format)
    img_dim = (256,256,3) # Manual entry

 

    try:
         print("Ok before directory this point")
         generator_model=CreatErrorMapModel(input_shape=img_dim,lastLayerActivation=lastLayerActivation, PercentageOfTrianable=PercentageOfTrianable, bnAtTheend=bnAtTheend,lossFunction=lossFunction)
         print("Ok before directory this point")
#-------------------------------------------------------------------------------
         logpath=os.path.join('../../log','DepthMapWith'+lastLayerActivation+str(PercentageOfTrianable)+'UnTr'+SpecificPathStr)
         modelPath=os.path.join('../../models','DepthMapwith'+lastLayerActivation+str(PercentageOfTrianable)+'Untr'+SpecificPathStr)
         shutil.rmtree(logpath, ignore_errors=True)
         shutil.rmtree(modelPath, ignore_errors=True)
         os.makedirs(logpath, exist_ok=True)
         os.makedirs(modelPath, exist_ok=True)
         print("Ok until this point")

#-----------------------PreTraining Depth Map-------------------------------------
         batchSize=batch_size
         history=generator_model.fit_generator(data_utils.facades_generator(img_dim,batch_size=batch_size), samples_per_epoch=nb_train_samples,epochs=epochs,verbose=1,validation_data=data_utils.facades_generator(img_dim,batch_size=batch_size),nb_val_samples=nb_validation_samples,callbacks=[
         keras.callbacks.ModelCheckpoint(os.path.join(modelPath,'DepthMap_weightsBestLoss.h5'), monitor='val_loss', verbose=1, save_best_only=True),
         keras.callbacks.ModelCheckpoint(os.path.join(modelPath,'DepthMap_weightsBestAcc.h5'), monitor='acc', verbose=1, save_best_only=True),
         keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
         keras.callbacks.TensorBoard(log_dir=logpath, histogram_freq=0, batch_size=batchSize, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)],)
         ErrorMap_weights_path = os.path.join(modelPath,'DepthMap_weights.h5' )
         generator_model.save_weights(ErrorMap_weights_path, overwrite=True)
         plt.plot(history.history['loss'])
         plt.savefig(logpath+"/history.png",bbox_inches='tight')
#------------------------------------------------------------------------------------
    except KeyboardInterrupt:
        pass
