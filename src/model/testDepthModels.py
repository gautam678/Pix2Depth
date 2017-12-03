from models import generator_unet_upsampling
import h5py
img_dim = (256,256,3)
bn_mode = 2
batch_size = 1
model_name = "generator_unet_upsampling"

import cv2
import sys
import numpy as np
from ErrorMapModel import CreatErrorMapModel
import argparse
import os
import shutil
import argparse

def facades_generator(img_dim_tuple,batch_size=10):
    path = "../../data/nyu_depth_v2_labeled.mat"
    f = h5py.File(path)
    img_dim = img_dim_tuple[0]
    while True:
        samples = np.random.choice(1449,batch_size,replace=True)
        img_batch = np.empty((batch_size,img_dim,img_dim,3))
        depth_batch = np.empty((batch_size,img_dim,img_dim,3))
        for index,i in enumerate(samples):
            img = f['images'][i]
            depth = f['depths'][i]
            img_=np.empty([img_dim,img_dim,3])
            img_[:,:,0] = cv2.resize(img[0,:,:].T,(img_dim,img_dim))
            img_[:,:,1] = cv2.resize(img[1,:,:].T,(img_dim,img_dim))
            img_[:,:,2] = cv2.resize(img[2,:,:].T,(img_dim,img_dim))
            depth_ = np.empty([img_dim, img_dim, 3])
            depth_[:,:,0] = cv2.resize(depth[:,:].T,(img_dim,img_dim))
            depth_[:,:,1] = cv2.resize(depth[:,:].T,(img_dim,img_dim))
            depth_[:,:,2] = cv2.resize(depth[:,:].T,(img_dim,img_dim))
            img_ = img_/255.0
            depth_ = depth_/10.0
            img_batch[index] = img_
            depth_batch[index] = depth_
        yield img_batch,depth_batch


gen = facades_generator(img_dim, batch_size=1)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ModelName', type=str, help="Model Name without postfix")
parser.add_argument('--ModelPath', type=str, help="PathStr")
parser.add_argument('--ModelPostfix', type=str, default=".h5", help="postfix of the model weights")
parser.add_argument('--NumberOfTestImages', type=int, default=20, help="Percentage of Triantable Layers")
args=parser.parse_args()

model=CreatErrorMapModel(input_shape=img_dim)
model.load_weights('../../models/'+args.ModelPath+'/'+ args.ModelName+args.ModelPostfix)

imagesPath=os.path.join('../../testResults',args.ModelPath,args.ModelName)
shutil.rmtree(imagesPath, ignore_errors=True)
os.makedirs(imagesPath, exist_ok=True)

for i in range(args.NumberOfTestImages):
	img, dep = next(gen)
#	cv2.imwrite("real.jpg",img[0]*255)
	dmap = model.predict(img)[0]
#	dmap[:][0] =dmap[:][1]=dmap[:][2]= (model.predict(img)[0][:][0]+model.predict(img)[0][:][1]+model.predict(img)[0][:][2])/3
	print ("depth:",dep[0][0][0])
	print (dep.shape)
	print ("predicted depth:",dmap[0][0])
	print("image:",img[0][0][0])
	cv2.imwrite( imagesPath+"/test_%d.jpg" % i,np.hstack(( img[0]*255,np.multiply(dmap,dmap)*125,dep[0]*255)))
#	cv2.imwrite( imagesPath+"/test_%d.jpg" % i,np.hstack(( img[0]*255,dmap*255,dep[0]*255)))

print ("Image to Depth is calculated here")
