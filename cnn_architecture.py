### Trying out a simple CNN autoencoder for depth estimation

import cv2
import numpy as np
from glob import glob
import sys
from keras.optimizers import SGD
from keras.layers import Conv2D, Input, MaxPooling2D as Pool, BatchNormalization as BN, UpSampling2D

rgbImagePaths = sorted(glob('imgs/*'))
depImagePaths = sorted(glob('deps/*'))

rgbPath = 'imgs/'
depPath = 'deps/'

def read_image(i):
    rPath = rgbPath+'img_%d.jpg' % i
    dPath = depPath+'dep_%d.jpg' % i
    return cv2.imread(rPath)/255., np.expand_dims(cv2.imread(dPath, 0),-1)/255.

images = map(read_image, range(1449))
X, Y = map(np.array, zip(*images))

from keras.models import Model
UpSampling2D((2,2))

def model_1():
    input_layer = Input(shape=(224,224,3))
    conv_1_a = Conv2D(8, 3, activation="relu", padding="same")(input_layer)
    conv_1_b = Conv2D(8, 3, activation="relu", padding="same")(conv_1_a)
    pool_1 = Pool((2,2))(conv_1_b)

    conv_2_a = Conv2D(16, 3, activation="relu", padding="same")(pool_1)
    conv_2_b = Conv2D(16, 3, activation="relu", padding="same")(conv_2_a)
    pool_2 = Pool((2,2))(conv_2_b)

    bn = BN()(pool_2)

    conv_3_a = Conv2D(32, 3, activation="relu", padding="same")(bn)
    conv_3_b = Conv2D(32, 3, activation="relu", padding="same")(conv_3_a)
    conv_3_c = Conv2D(32, 3, activation="relu", padding="same")(conv_3_b)
    pool_3 = Pool((2,2))(conv_3_c)

    up_1 = UpSampling2D((2,2))(pool_3)
    conv_4_a = Conv2D(16, 3, activation="relu", padding="same")(up_1)
    conv_4_b = Conv2D(16, 3, activation="relu", padding="same")(conv_4_a)

    up_2 = UpSampling2D((2,2))(conv_4_a)
    conv_5_a = Conv2D(8, 3, activation="relu", padding="same")(up_2)
    conv_5_b = Conv2D(8, 3, activation="relu", padding="same")(conv_5_a)

    up_3 = UpSampling2D((2,2))(conv_5_b)
    conv_out = Conv2D(1, 3, activation="sigmoid", padding="same")(up_3)

    model = Model(inputs=input_layer,outputs=conv_out)

    return model

def model_2():

    input_layer = Input(shape=(224,224,3))
    conv = Conv2D(4, 3, activation="relu", padding="same")(input_layer)
    conv = BN()(conv)
    conv = Conv2D(8, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(8, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(16, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(16, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(8, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(8, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(4, 3, activation="relu", padding="same")(conv)
    conv = BN()(conv)
    conv = Conv2D(4, 3, activation="relu", padding="same")(conv)
    conv_out = Conv2D(1, 3, activation="sigmoid", padding="same")(conv)

    model = Model(inputs=input_layer,outputs=conv_out)

    return model

def model_3():

    input_layer = Input(shape=(224,224,3))
    from keras.applications.resnet50 import ResNet50, preprocess_input
    from keras.layers import Conv2DTranspose as DeConv
    resnet = ResNet50(include_top=False, weights="imagenet")
    resnet.trainable = False

    res_features = resnet(input_layer)

    conv = DeConv(1024, padding="valid", activation="relu", kernel_size=3)(res_features)
    conv = UpSampling2D((2,2))(conv)
    conv = DeConv(512, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2,2))(conv)
    conv = DeConv(128, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2,2))(conv)
    conv = DeConv(32, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2,2))(conv)
    conv = DeConv(8, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2,2))(conv)
    conv = DeConv(4, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = DeConv(1, padding="valid", activation="sigmoid", kernel_size=5)(conv)

    model = Model(inputs=input_layer, outputs=conv)
    return model

model_num = int(sys.argv[1])
model_name = ['hourglass','block','resglass'][model_num - 1]
if model_num == 1:
    model = model_1()
elif model_num == 2:
    model = model_2()
else:
    model = model_3()
model.summary()

print X.shape, Y.shape
print 'Training ...'

lr = 1.0
for i in range(2000):
    print i
    cv2.imwrite("preds/model_{}_ep_{}.jpg".format(model_name, i), model.predict(X[:1])[0]*255.)
    model.compile(loss="mse", optimizer=SGD(lr=lr, decay=1e-2))
    model.fit(X,Y, epochs=5)
    lr *= 0.95
    model.save_weights('weights_%s.h5' % model_name)
