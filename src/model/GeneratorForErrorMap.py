from models import generator_unet_upsampling
import h5py
img_dim = (256,256,3)
bn_mode = 2
batch_size = 32
model_name = "generator_unet_upsampling"
modelEpoch=130
threshold=0.1

import cv2
import sys
import numpy as np
from fromDatasetGenerator import  fDataSet_generator
def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 1  # -inf inf NaN
    return c

def GeneratorErrorMap(img_dim_tuple,batch_size=10, modelEpoch=130, threshold=0.1,f=None,indexes=1448):
   if(f==None):
   	path = "../../data/nyu_depth_v2_labeled.mat"
   	f = h5py.File(path)
   gen , length = fDataSet_generator(img_dim, batch_size,f=f)
   model = generator_unet_upsampling(img_dim, bn_mode, batch_size)
   model.load_weights('../../models/CNN/pix2depthgen_weights_epoch%d.h5' % modelEpoch) #gen_weights_epoch45.h5
   def GenFunction():
       while True:
           img, dep = gen.__next__()
           dmap = model.predict(img)
#           errorMap=(np.divide(np.fabs(dmap-dep), dep)<threshold)
#           errorMap=(np.divide(np.fabs(dmap-dep), dep)<threshold).astype(int)
#           errorMap=(div0(np.fabs(dmap-dep), dep)<threshold).astype(int)
           errorMap=(np.fabs(dmap-dep)<threshold).astype(int)
           yield (img,errorMap)
   return(GenFunction(),1448) #todo: accurate Value

if __name__ == "__main__":
	(ErrorMapGen,length) =GeneratorErrorMap(img_dim, batch_size=batch_size,modelEpoch=130, threshold=0.1)
	(img, ErrorMap) = ErrorMapGen.__next__()
	print(ErrorMap)
	print(ErrorMap.shape)
