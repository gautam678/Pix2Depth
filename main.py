#from src.model.models import generator_unet_upsampling
from PIL import Image
import h5py
import cv2
import sys
import os
import numpy as np
import cv2
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from config import CONFIG
from inf_cycle import test, test_dep

img_dim = 224
output_path = 'static/results'

# Make dictionary
if not CONFIG['development']:
    # print 'Loading p2d_model'
    # p2d_model = load_model('weights/model_resglass.h5')

    # print 'Loading d2p_model'
    # d2p_model = load_model('weights/model_resglass.h5')
    model_list = {  
            'pix2depth':{ 
                'pix2pix' : load_model('weights/p2d_pix2pix.h5'),
                'CycleGAN':load_model('weights/p2d_cycle.h5'),
                'CNN': load_model('weights/p2d_cnn.h5'),
                },
            'depth2pix':{ 
                'pix2pix' : load_model('weights/d2p_pix2pix.h5'),
                'CycleGAN':load_model('weights/d2p_cycle.h5'),
                }
             }

def pix2depth(path, model):
    model_name = 'p2d'
    originalImage = cv2.imread(path)
    loaded_model =  model_list['pix2depth'][model]
    file_name = model+'_'+path.split('/')[-1]
    output_file = os.path.join(output_path,file_name)
    if model =='CNN':
        originalImage = cv2.resize(originalImage,(img_dim,img_dim))
        x = preprocess_input(originalImage/1.)
    elif model == 'CycleGAN':
        test(path)
        os.system('cp gautam/inf_results/imgs/fakeA_0_0.jpg %s' % output_file)
    else:
        originalImage = cv2.resize(originalImage,(256,256))
        x = originalImage/255.
    if not model == 'CycleGAN':
        p1 = get_depth_map(x, loaded_model)
        cv2.imwrite(output_file,p1)
    return output_file

def depth2pix(path,model):
    model_name = 'd2p'
    originalImage = cv2.imread(path)
    loaded_model =  model_list['depth2pix'][model]
    file_name = model+'_'+path.split('/')[-1]
    output_file = os.path.join(output_path,file_name)
    if model =='CNN':
        img_dim = 256
        originalImage = cv2.resize(originalImage,(img_dim,img_dim))
        x = preprocess_input(originalImage/1.)
    elif model == 'CycleGAN':
        test_dep(path)
        os.system('cp gautam/inf_results/imgs/fakeB_0_0.jpg %s' % output_file)
    else:
        originalImage = cv2.resize(originalImage,(256,256))
        x = originalImage/255.
    if not model == 'CycleGAN':
        p1 = get_depth_map(x, loaded_model)
        cv2.imwrite(output_file,p1)
    return output_file

def blur_effect(image, depthImage, outputPath):
    try:
        if len(depthImage.shape) == 3:
            depthImage = np.mean(depthImage, axis=-1).astype(int)
        print depthImage.shape
        (h, w) = depthImage.shape
        image = cv2.resize(image, (h,w))
        blurredImage = cv2.GaussianBlur(image,(5,5),0)
        print 'b'
        # Need path to depth Image
        thresh = 200
        maskImage = cv2.threshold(depthImage, thresh, 255, cv2.THRESH_BINARY)[1]
        print maskImage.shape
        print h,w
        new_image = np.zeros((h, w, 3),dtype=np.int)
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

def portrait_mode(path, model):
    originalImage = cv2.imread(path)
    file_name = model+'_'+path.split('/')[-1]
    output_file = pix2depth(path, model)

    portrait_out_path = os.path.join(output_path, 'portrait_'+file_name)
    p1 = cv2.imread(output_file, 0)
    if blur_effect(originalImage, p1, portrait_out_path):
        return portrait_out_path 
