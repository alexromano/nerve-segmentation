import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import math
import cv2

files = os.listdir("ultrasound-nerve-segmentation/train")
image_names = []
for f in files:
    if '_mask' not in f:
        image_names.append(os.path.splitext(f)[0])
        
train_names, test_names = train_test_split(image_names, test_size=0.1)
train_len = len(train_names)
test_len = len(test_names)

ORIGINAL_WIDTH = 420
ORIGINAL_HEIGHT = 580
IMG_WIDTH = 128
IMG_HEIGHT = 128

# a smoothed version of dice coefficient
def dice(y, y_pred):
    intersection = K.sum(K.flatten(y) * K.flatten(y_pred))
    return (2.0 * intersection + 1) / (K.sum(y) + K.sum(y_pred) + 1)

def dice_loss(y, y_pred):
    return -tf.log(dice(y, y_pred))

# Build U-Net/FCN style model
def build_net(image_width, image_height, batch_size, learning_rate):
    input_images = Input(shape=(image_width, image_height, 1), name='input')
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(input_images)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(2)(conv1)

    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(2)(conv2)

    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(2)(conv3)

    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(2)(conv4)

    conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    conv6 = concatenate([Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv5), conv4])
    drop6 = Dropout(rate=0.5)(conv6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(drop6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)

    conv7 = concatenate([Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv6), conv3])
    drop7 = Dropout(rate=0.5)(conv7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(drop7)
    # conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)

    conv8 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv7), conv2])
    drop8 = Dropout(rate=0.5)(conv8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(drop8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)

    conv9 = concatenate([Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv8), conv1])
    drop9 = Dropout(rate=0.5)(conv9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(drop9)
    # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)

    output = Conv2D(1, (1,1), activation='sigmoid')(conv9)

    model = Model(inputs=[input_images], outputs=[output])
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_loss, metrics=[dice])
    
    return model

def preprocess(im):
    im_arr = np.asarray(im.resize((IMG_WIDTH, IMG_HEIGHT)))
    return im_arr

def augment(img, label):
    #flip h
    flipped_h = np.flip(img, axis=0)
    flipped_h_lb = np.flip(label, axis=0)
    #flip v
    flipped_v = np.flip(img, axis=1)
    flipped_v_lb = np.flip(label, axis=1)
    #rotate
    rotate_90 = np.rot90(img)
    rotate_90_lb = np.rot90(label)
    rotate_180 = np.rot90(rotate_90)
    rotate_180_lb = np.rot90(rotate_90_lb)
    rotate_270 = np.rot90(rotate_180)
    rotate_270_lb = np.rot90(rotate_180_lb)
    aug_ims = np.array([flipped_h, flipped_v, rotate_90, rotate_180, rotate_270])
    aug_lbs = np.array([flipped_h_lb, flipped_v_lb, rotate_90_lb, rotate_180_lb, rotate_270_lb])
    return (aug_ims, aug_lbs)

def get_batches(batch_size, train=True):
    filenames = None
    if train: filenames = train_names
    else: filenames = test_names
    while True:
        for i in range(0, len(filenames), batch_size):
            images = []
            labels = []
            for img_name in filenames[i:i+batch_size]:
                im = Image.open("ultrasound-nerve-segmentation/train/"+img_name+".tif")
                lb = Image.open("ultrasound-nerve-segmentation/train/"+img_name+"_mask.tif")
                im_arr = preprocess(im)
                lb_arr = preprocess(lb)
                images.append(im_arr)
                labels.append(lb_arr)
                
                if train:
                    aug_ims, aug_lbs = augment(im_arr, lb_arr)
                    for im in aug_ims:
                        images.append(im)
                    for lb in aug_lbs:
                        labels.append(lb)
            ims = np.expand_dims(np.array(images, dtype='float32'), axis=3)
            lbls = np.expand_dims(np.array(labels, dtype='float32'), axis=3)

            mean = np.mean(ims)
            std = np.std(ims)
            ims -= mean
            ims /= std
            lbls /= 255.0

            yield ims, lbls
    
def get_train_data():
    print("loading training images")
    files = np.array(os.listdir("ultrasound-nerve-segmentation/train"))
    image_names = files[np.where(np.char.find(files, '_mask')<0)]
    splitfile = np.vectorize(lambda x: os.path.splitext(x)[0])
    image_names = splitfile(image_names)
    
    train_images = np.zeros((image_names.shape[0], IMG_HEIGHT, IMG_WIDTH))
    train_masks = np.zeros((image_names.shape[0], IMG_HEIGHT, IMG_WIDTH))
    for i in range(len(train_names)):
        im = Image.open("ultrasound-nerve-segmentation/train/"+train_names[i]+".tif")
        mask = Image.open("ultrasound-nerve-segmentation/train/"+train_names[i]+"_mask.tif")
        # TODO: try chaning this to cv2.resize and see?
        im_arr = np.array(im.resize((IMG_WIDTH, IMG_HEIGHT)))
        mask_arr = np.array(mask.resize((IMG_WIDTH, IMG_HEIGHT)))
        train_images[i] = im_arr
        train_masks[i] = mask_arr
    
    mean = np.mean(train_images)
    std = np.std(train_images)
    train_images -= mean
    train_images /= std
    train_masks /= 255.0 
    return np.expand_dims(train_images, axis=3), np.expand_dims(train_masks, axis=3)
    
def train(learning_rate, epochs, batch_size):
    
    # gen = get_batches(batch_size)
    # val = get_batches(batch_size, train=False)
    train_images, train_masks = get_train_data()
    model = build_net(IMG_WIDTH, IMG_HEIGHT, batch_size, learning_rate)
    checkpoint = ModelCheckpoint('model_weights_zscored.hd5', monitor='val_loss')
    # model.fit_generator(gen, epochs=epochs, steps_per_epoch=int(math.ceil(train_len/batch_size)),
    #                     validation_data=val, validation_steps=int(math.ceil(test_len/batch_size)), 
    #                     verbose=1, callbacks=[checkpoint])
    model.fit(train_images, train_masks, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], 
        validation_split=0.1)

if __name__ == '__main__': train(1e-3, 50, 32)
