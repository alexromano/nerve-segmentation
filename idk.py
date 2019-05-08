import numpy as np
from datetime import datetime as dt
from PIL import Image
import os
import csv
import cv2
from tensorflow.keras.models import Model, load_model
from train import preprocess, dice, LossHistory, get_train_data()
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# one possible issue: some loss is happening on the resize
# another issue: which is the width and which is the height?
#     checked, seems fine
# load some images from the train set
# resize them and run through the network
# size them back up and get dice loss with the masks

original_width = 580
original_height = 420
img_width = 128
img_height = 128

def eval(image_names):
        # get dice coeff
        print("loading model and predicting")
        # load model and predict on images
        model = load_model('model_weights_overfit.hd5', compile=False)
        img_masks = model.predict(test_imgs, batch_size=3, verbose=1)
        # resize
        masks_resized = np.zeros((img_masks.shape[0], original_height, original_width))
        for i in range(img_masks.shape[0]):
        im_resized = np.array(cv2.resize(img_masks[i], (original_width, original_height)))
        masks_resized[i] = im_resized
        print("Loading ground truth and dicing")
        # load ground truth masks for these images
        labels = np.zeros((len(test_imgs), original_height, original_width))
        for f in image_names[:10]:
                im = Image.open("ultrasound-nerve-segmentation/masks/masks/"+f+ "_mask.tif")
                labels[i] = np.array(im)
        # plt.subplot(521)
        # for i in masks_resized:
        #         plt.imshow(i)
        # plt.show()
        # get dice between ground truth and predicted
        d = dice(labels, masks_resized)
        import tensorflow as tf
        print(tf.Session().run(d))

def main():
        data = get_train_data()
        
