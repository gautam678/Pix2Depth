from data_utils import load_data,facades_generator
import cv2

x_batch,y_batch = next(facades_generator(256,10))
x_train,y_train = load_data("channels_last")
print x_train.shape