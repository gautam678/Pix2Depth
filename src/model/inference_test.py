from models import generator_unet_upsampling
import h5py
img_dim = (256,256,3)
bn_mode = 2
batch_size = 1
model_name = "generator_unet_upsampling"

import cv2
import sys
import numpy as np

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

img, dep = next(gen)
cv2.imwrite("real.jpg",img[0]*255)
for i in range(0,200,5):
	model = generator_unet_upsampling(img_dim, bn_mode, batch_size)
	model.load_weights('../../models/CNN/gen_weights_epoch%d.h5' % i) #gen_weights_epoch45.h5
	dmap = model.predict(dep)[0]
	print dep[0][0][0]
	print dep.shape
	print dmap[0][0][0]
	cv2.imwrite("test_%d.jpg" % i,np.hstack((dmap*255,dep[0]*255)))
