import numpy as np
from datetime import datetime as dt
from PIL import Image
import os
import csv
import cv2
from tensorflow.keras.models import Model, load_model
from train import preprocess, dice
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

print("loading test images")
files = os.listdir("ultrasound-nerve-segmentation/train")
image_names = []
for f in files[90:115]:
    if '_mask' not in f:
        image_names.append(os.path.splitext(f)[0])  
train_names, test_names = train_test_split(image_names, test_size=0.1)
test_imgs = []
for f in test_names:
    if "_mask" not in f:
        im = Image.open("ultrasound-nerve-segmentation/train/"+f+".tif")
        # TODO: try chaning this to cv2.resize and see?
        im_arr = np.asarray(im.resize((img_width, img_height)))
        test_imgs.append(im_arr)
mean = np.mean(test_imgs)
std = np.std(test_imgs)
test_imgs -= mean
test_imgs /= std
test_imgs = np.expand_dims(np.array(test_imgs, dtype='float32'), axis=3)

# label_names = os.listdir('ultrasound-nerve-segmentation/labels')
# labels = []
# for l in label_names:
#    im = Image.open("ultrasound-nerve-segmentation/test/"+f)
#    im_arr = np.asarray(im.resize((img_width, img_height)))
#    test_imgs.append(im_arr)

# get dice coeff
print("loading model and predicting")
# load model and predict on images
model = load_model('model_weights.hd5', compile=False)
img_masks = model.predict(test_imgs, batch_size=256, verbose=1)
# resize
masks_resized = np.zeros((img_masks.shape[0], original_height, original_width))
for i in range(img_masks.shape[0]):
    im_resized = np.array(cv2.resize(img_masks[i], (original_width, original_height)))
    masks_resized[i] = im_resized
print("Loading ground truth and dicing")
# load ground truth masks for these images
labels = np.zeros((len(test_names), original_height, original_width))
for f in test_names:
    if "_mask" in f:
        im = Image.open("ultrasound-nerve-segmentation/train/"+f+ "_mask.tif")
        labels[i] = np.array(im)
# plt.subplot(521)
# for i in masks_resized:
#         plt.imshow(i)
# plt.show()
# get dice between ground truth and predicted
d = dice(labels, masks_resized)
import tensorflow as tf
print(tf.Session().run(d))
