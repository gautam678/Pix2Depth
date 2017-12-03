from src.model.models import generator_unet_upsampling
import h5py
import cv2
import sys
import os
import numpy as np

img_dim = 256
bn_mode = 2
batch_size = 1
model_name = "generator_unet_upsampling"
model = generator_unet_upsampling((256,256,3), bn_mode, batch_size)
model.load_weights('weights/gen_weights_epoch125.h5')
output_path = 'static/results'

def process_rgb(path):
    rgb_image = cv2.imread(path)
    img=np.empty([img_dim,img_dim,3])
    img[:,:,0] = cv2.resize(rgb_image[2,:,:].T,(img_dim,img_dim))
    img[:,:,1] = cv2.resize(rgb_image[1,:,:].T,(img_dim,img_dim))
    img[:,:,2] = cv2.resize(rgb_image[0,:,:].T,(img_dim,img_dim))
    img = np.expand_dims(img, axis=0)
    rgb = model.predict(img)[0]
    output_file = os.path.join(os.path.join(output_path,'test.jpg'))
    cv2.imwrite(output_file,rgb)
    return output_file

def process(image):
    image = cv2.GaussianBlur(image,(5,5),0)
    return image

def mask(image):
    image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    thresh = 150
    im_bw = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    return im_bw

def portrait_mode(path,outputPath):
    image = cv2.imread(path)
    blurredImage = process(image)
    # Need path to depth Image
    depthImage = cv2.imread(outputPath)
    maskImage = mask(depthImage)
    new_image = np.zeros((224,224,3),dtype=np.int)
    for i in range(len(maskImage)):
        for j in range(len(maskImage[i])):
            if maskImage[i][j] ==255.0:
                new_image[i,j,0] = blurredImage[i,j,0]
                new_image[i,j,1] = blurredImage[i,j,1]
                new_image[i,j,2] = blurredImage[i,j,2]
            else:
                new_image[i,j,0] = image[i,j,0]
                new_image[i,j,1] = image[i,j,1]
                new_image[i,j,2] = image[i,j,2]
    cv2.imwrite(outputPath+'/potrait.jpg', new_image)
    return os.path.join(outputPath,'potrait.jpg')

