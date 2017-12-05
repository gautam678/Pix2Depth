## test inference

import cv2
from cnn_architecture import model_1, model_2, model_3, preprocess_input, images
from keras.models import load_model
import numpy as np


def get_depth_map(input_image, model):
    pred_dep = model.predict(np.array([input_image]), batch_size=1)[0]*255.
    return pred_dep

def test_1():
    _id = 9
    originalImage = cv2.imread('imgs/img_%d.jpg' % _id)
    x = preprocess_input(originalImage/1.)
    cv2.imwrite('_testImage.jpg', x)
    y = images[_id][1]
    cv2.imwrite('_testDep.jpg',y*255.)
    model = model_1()
    model.load_weights('weights_hourglass.h5')
    p1 = get_depth_map(x, model)
    cv2.imwrite('_p1.jpg', p1)

    model = model_3()
    #model = load_model('model_hourglass.h5')
    model.load_weights('weights_resglass.h5')
    p3 = get_depth_map(x, model)
    cv2.imwrite('_p3.jpg',p3)

    y = cv2.imread('_testDep.jpg')
    p1 = cv2.imread('_p1.jpg')
    p3 = cv2.imread('_p3.jpg')
    cv2.imwrite('tests/test_%d.jpg' % _id,np.hstack((originalImage,y,p1,p3)))

test_1()

def web_get_depth(imagePath, outPath):
    return
