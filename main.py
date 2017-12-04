#from src.model.models import generator_unet_upsampling
import h5py
import cv2
import sys
import os
import numpy as np
import cv2
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model

img_dim = 224
output_path = 'static/results'

#p2d_model = load_model('weights/siva.h5')
#d2p_model = load_model('weights/marzi.h5')

# First Page
def pix2depth(path):
    originalImage = cv2.imread(path)
    originalImage = cv2.resize(originalImage,(img_dim,img_dim))
    x = preprocess_input(originalImage/1.)
    model = load_model('weights/model_resglass.h5' % model_name)
    p1 = get_depth_map(x, model)
    file_name = model_name+'_'+path.split('/')[-1]
    output_file = os.path.join(output_path,file_name)
    cv2.imwrite(output_file,p1)
    return output_file

def depth2pix(path):
    originalImage = cv2.imread(path)
    originalImage = cv2.resize(originalImage,(img_dim,img_dim))
    x = preprocess_input(originalImage/1.)
    model = load_model('weights/model_resglass.h5' % model_name)
    p1 = get_depth_map(x, model)
    file_name = model_name+'_'+path.split('/')[-1]
    output_file = os.path.join(output_path,file_name)
    cv2.imwrite(output_file,p1)
    return output_file

def process(image):
    image = cv2.GaussianBlur(image,(5,5),0)
    return image

def mask(image):
    image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    thresh = 150
    im_bw = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    return im_bw

def portrait_mode(image, depthImage, outputPath):
    try:
        blurredImage = cv2.GaussianBlur(image,(5,5),0)
        # Need path to depth Image
        thresh = 100 
        maskImage = cv2.threshold(depthImage, thresh, 255, cv2.THRESH_BINARY)[1]
        print maskImage.shape
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
        cv2.imwrite(outputPath, new_image)
        return True
    except Exception as e:
        print e
        return False


def get_depth_map(input_image, model):
    pred_dep = model.predict(np.array([input_image]), batch_size=1)[0]*255.
    return pred_dep

# Potrait Mode
def portrait_mode(path, model_name="siva"):
    originalImage = cv2.imread(path)
    img_dim = 256 if model_name != 'siva' else 224
    originalImage = cv2.resize(originalImage,(img_dim,img_dim))
    x = preprocess_input(originalImage/1.)
    model = load_model('weights/model_resglass.h5' % model_name)
    model.summary()
    p1 = get_depth_map(x, model)
    file_name = model_name+'_'+path.split('/')[-1]
    output_file = os.path.join(output_path,file_name)
    cv2.imwrite(output_file,p1)
    portrait_out_path = os.path.join(output_path, 'portrait_'+file_name)
    if portrait_mode(originalImage, p1, portrait_out_path):
        return portrait_out_path 
