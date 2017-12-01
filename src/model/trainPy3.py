import os
import sys
import time
import numpy as np
import models
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
# Utils
sys.path.append("../utils")
import general_utils
import data_utils

from ErrorMapModel import CreatErrorMapModel

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    model_name = kwargs["model_name"]
    generator = kwargs["generator"]
    image_data_format = kwargs["image_data_format"]
    img_dim = kwargs["img_dim"]
    patch_size = kwargs["patch_size"]
    bn_mode = kwargs["bn_mode"]
    label_smoothing = kwargs["use_label_smoothing"]
    label_flipping = kwargs["label_flipping"]
    dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]
    lastLayerActivation=kwargs["lastLayerActivation"]
    PercentageOfTrianable=kwargs["PercentageOfTrianable"]
    SpecificPathStr=kwargs["SpecificPathStr"]
    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    #general_utils.setup_logging(model_name)

    # Load and rescale data
    #X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format)
    img_dim = (256,256,3) # Manual entry

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        """
        generator_model = models.load("generator_unet_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size)
        """
        generator_model=CreatErrorMapModel(input_shape=img_dim,lastLayerActivation=lastLayerActivation, PercentageOfTrianable=PercentageOfTrianable)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator",
                                          img_dim_disc,
                                          nb_patch,
                                          bn_mode,
                                          use_mbd,
                                          batch_size)

         generator_model.compile(loss='mae', optimizer=opt_discriminator)
#-------------------------------------------------------------------------------
         logpath=os.path.join('../../log','DepthMapWith'+lastLayerActivation+str(PercentageOfTrianable)+'UnTr'+SpecificPathStr)
         modelPath=os.path.join('../../models','DepthMapwith'+lastLayerActivation+str(PercentageOfTrianable)+'Untr'+SpecificPathStr)
         os.makedirs(logpath, exist_ok=True)
         os.makedirs(modelPath, exist_ok=True)os.makedirs(modelPath, exist_ok=True)

#-----------------------PreTraining Depth Map-------------------------------------
         nb_train_samples = 2000
         nb_validation_samples = 
         epochs = 20
         history=whole_model.fit_generator(data_utils.facades_generator(img_dim,batch_size=batch_size), samples_per_epoch=nb_train_samples,epochs=epochs,validation_data=data_utils.facades_generator(img_dim,batch_size=batch_size),nb_val_samples=nb_validation_    samples,       callbacks=[
         keras.callbacks.ModelCheckpoint(os.path.join(modelPath,'DepthMap_weightsBestLoss.h5'), monitor='val_loss', verbose=1, save_best_only=True),
         keras.callbacks.ModelCheckpoint(os.path.join(modelPath,'DepthMap_weightsBestAcc.h5'), monitor='acc', verbose=1, save_best_only=True),
         keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
         keras.callbacks.TensorBoard(log_dir=logpath, histogram_freq=0, batch_size=batchSize, write_graph=True, write_grads=False, write_images=True, embeddin    gs_freq=0, embeddings_layer_names=None, embeddings_metadata=None)],)
#------------------------------------------------------------------------------------


        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   img_dim,
                                   patch_size,
                                   image_data_format)

        loss = [l1_loss, 'binary_crossentropy']
        loss_weights = [1E1, 1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        discriminator_model.trainable = True
        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        gen_loss = 100
        disc_loss = 100

        # Start training
        print("Start training")
        for e in range(nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            for X_full_batch, X_sketch_batch in data_utils.facades_generator(img_dim,batch_size=batch_size):

                # Create a batch to feed the discriminator model
                X_disc, y_disc = data_utils.get_disc_batch(X_full_batch,
                                                           X_sketch_batch,
                                                           generator_model,
                                                           batch_counter,
                                                           patch_size,
                                                           image_data_format,
                                                           label_smoothing=label_smoothing,
                                                           label_flipping=label_flipping)

                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, y_disc) # X_disc, y_disc
                # Create a batch to feed the generator model
                X_gen_target, X_gen = next(data_utils.facades_generator(img_dim,batch_size=batch_size))
                y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                y_gen[:, 1] = 1

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                batch_counter += 1
                progbar.add(batch_size, values=[("D logloss", disc_loss),
                                                ("G tot", gen_loss[0]),
                                                ("G L1", gen_loss[1]),
                                                ("G logloss", gen_loss[2])])

                # Save images for visualization
                if batch_counter % (n_batch_per_epoch / 2) == 0:
                    # Get new images from validation
                    figure_name = "training_"+str(e)
                    data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                    batch_size, image_data_format, figure_name)

                if batch_counter >= n_batch_per_epoch:
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

            if e % 5 == 0:
                gen_weights_path = os.path.join('../../models/%s/gen_weights_epoch%s.h5' % (model_name, e))
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join('../../models/%s/disc_weights_epoch%s.h5' % (model_name, e))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join('../../models/%s/DCGAN_weights_epoch%s.h5' % (model_name, e))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

    except KeyboardInterrupt:
        pass
