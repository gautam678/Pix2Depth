import h5py
import numpy as np
import cv2
from PIL import Image
path = 'nyu_depth_v2_labeled.mat'
f = h5py.File(path)
print len(f['images'])
for i in range(0,1449):
    img = f['images'][i]
    depth = f['depths'][i]
    img_=np.empty([480,640,3])
    img_[:,:,0] = img[0,:,:].T
    img_[:,:,1] = img[1,:,:].T
    img_[:,:,2] = img[2,:,:].T
    depth_ = np.empty([480, 640, 3])
    depth_[:,:,0] = depth[:,:].T
    depth_[:,:,1] = depth[:,:].T
    depth_[:,:,2] = depth[:,:].T
    print depth_.shape
    print img_.shape
    img_ = img_/255.0
    depth_ = depth_/4.0
    both = np.hstack((img_,depth_))
    cv2.imshow('Demo 2 Image',both)
    k = cv2.waitKey(0)
    if k==27:    # Esc key to stop
        break
    else:
        continue
    cv2.destroyAllWindows()
