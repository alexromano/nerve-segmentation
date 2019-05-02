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
from keras.preprocessing.image import ImageDataGenerator
import math
import cv2

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
    #drop6 = Dropout(rate=0.5)(conv6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)

    conv7 = concatenate([Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv6), conv3])
    #drop7 = Dropout(rate=0.5)(conv7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)

    conv8 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv7), conv2])
    #drop8 = Dropout(rate=0.5)(conv8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)

    conv9 = concatenate([Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv8), conv1])
    #drop9 = Dropout(rate=0.5)(conv9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)

    output = Conv2D(1, (1,1), activation='sigmoid')(conv9)

    model = Model(inputs=[input_images], outputs=[output])
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_loss, metrics=[dice])
    
    return model
    
def get_train_data():
    print("loading training images")
    files = np.array(os.listdir("ultrasound-nerve-segmentation/train/train"))
    # image_names = files[np.where(np.char.find(files, '_mask')<0)]
    splitfile = np.vectorize(lambda x: os.path.splitext(x)[0])
    image_names = splitfile(files)
    
    train_images = np.zeros((image_names.shape[0], IMG_HEIGHT, IMG_WIDTH))
#    train_masks = np.zeros((image_names.shape[0], IMG_HEIGHT, IMG_WIDTH))
    for i in range(len(image_names)):
        im = Image.open("ultrasound-nerve-segmentation/train/train/"+image_names[i]+".tif")
 #       mask = Image.open("ultrasound-nerve-segmentation/masks/masks/"+image_names[i]+"_mask.tif")
        # TODO: try chaning this to cv2.resize and see?
        im_arr = np.array(im.resize((IMG_WIDTH, IMG_HEIGHT)))
  #      mask_arr = np.array(mask.resize((IMG_WIDTH, IMG_HEIGHT)))
        train_images[i] = im_arr
   #     train_masks[i] = mask_arr
    
   # mean = np.mean(train_images)
   # std = np.std(train_images)
   # train_images -= mean
   # train_images /= std
   # train_masks /= 255.0 
    return np.expand_dims(train_images, axis=3)#, np.expand_dims(train_masks, axis=3)
    
def train(learning_rate, epochs, batch_size):
    train_images = get_train_data()
    train_len = len(train_images)

    image_datagen = ImageDataGenerator(featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.15)

    mask_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.15)

    seed = 1
    image_datagen.fit(train_images, augment=True, seed=seed)
    #mask_datagen.fit(train_masks, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        'ultrasound-nerve-segmentation/train',
        target_size=(128,128),
        color_mode='grayscale',
        class_mode=None,
        seed=seed, subset='training')

    mask_generator = mask_datagen.flow_from_directory(
        'ultrasound-nerve-segmentation/masks',
        target_size=(128,128),
        color_mode='grayscale',
        class_mode=None,
        seed=seed, subset='training')

    image_val_gen = image_datagen.flow_from_directory(
        'ultrasound-nerve-segmentation/train',
        target_size=(128,128),
        color_mode='grayscale',
        class_mode=None,
        seed=seed, subset='validation') 

    mask_val_gen = mask_datagen.flow_from_directory(
        'ultrasound-nerve-segmentation/masks',
        target_size=(128,128),
        color_mode='grayscale',
        class_mode=None,
        seed=seed, subset='validation') 
    print('zipping')
    # combine generators into one which yields image and masks
    train_generator = combine_generator(image_generator, mask_generator)
    print('zip2')
    validation_generator = combine_generator(image_val_gen, mask_val_gen)
    
    print('training')
    model = build_net(IMG_WIDTH, IMG_HEIGHT, batch_size, learning_rate)
    checkpoint = ModelCheckpoint('model_weights_zscored.hd5', monitor='val_loss')
    model.fit_generator(
        train_generator,
        steps_per_epoch=149,
        epochs=50,
        validation_data=validation_generator, validation_steps=26,
        verbose=1, callbacks=[checkpoint])

def combine_generator(gen1, gen2):
    while True:
         yield(gen1.next(), gen2.next()) 
                    
if __name__ == '__main__': train(1e-4, 50, 32)
