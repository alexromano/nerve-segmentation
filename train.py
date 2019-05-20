import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, BatchNormalization, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
import wandb
from wandb.keras import WandbCallback
import math
import json
import cv2

ORIGINAL_WIDTH = 420
ORIGINAL_HEIGHT = 580
IMG_WIDTH = 128
IMG_HEIGHT = 128

# a smoothed version of dice coefficient
# a smoothed version of dice coefficient
def dice(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_loss(y_true, y_pred):
    return -dice(y_true, y_pred)

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
    conv6 = SpatialDropout2D(rate=0.5)(conv6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)
    #conv6 = BatchNormalization()(conv6)

    conv7 = concatenate([Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv6), conv3])
    conv7 = SpatialDropout2D(rate=0.5)(conv7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)
    #conv7 = BatchNormalization()(conv7)

    conv8 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv7), conv2])
    conv8 = SpatialDropout2D(rate=0.5)(conv8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)
    #conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)
    #conv8 = BatchNormalization()(conv8)

    conv9 = concatenate([Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv8), conv1])
    conv9 = SpatialDropout2D(rate=0.5)(conv9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    #conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    #conv9 = BatchNormalization()(conv9)

    output = Conv2D(1, (1,1), activation='sigmoid')(conv9)

    model = Model(inputs=[input_images], outputs=[output])
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_loss, metrics=[dice])
    
    return model
    
def get_train_data(image_names):
    print("loading training images")
    
    train_images = np.zeros((image_names.shape[0], IMG_HEIGHT, IMG_WIDTH))
    train_masks = np.zeros((image_names.shape[0], IMG_HEIGHT, IMG_WIDTH))
    for i in range(0, len(image_names)):
        im = Image.open("ultrasound-nerve-segmentation/train/train/"+image_names[i]+".tif")
        mask = Image.open("ultrasound-nerve-segmentation/masks/masks/"+image_names[i]+"_mask.tif")
        # TODO: try chaning this to cv2.resize and see?
        im_arr = np.array(im.resize((IMG_WIDTH, IMG_HEIGHT)))
        mask_arr = np.array(mask.resize((IMG_WIDTH, IMG_HEIGHT)))

        train_images[i] = im_arr
        train_masks[i] = mask_arr
        
    return np.expand_dims(train_images, axis=3), np.expand_dims(train_masks, axis=3)

def get_val_data():
    val_files = np.array(os.listdir("ultrasound-nerve-segmentation/val/imgs"))
    splitfile = np.vectorize(lambda x: os.path.splitext(x)[0])  
    val_names = splitfile(val_files)

    val_images = np.zeros((val_names.shape[0], IMG_HEIGHT, IMG_WIDTH))
    val_masks = np.zeros((val_names.shape[0], IMG_HEIGHT, IMG_WIDTH))
    for i in range(0, len(val_names)):
        im = Image.open("ultrasound-nerve-segmentation/val/imgs/"+val_names[i]+".tif")
        mask = Image.open("ultrasound-nerve-segmentation/val/masks/"+val_names[i]+"_mask.tif")
        im_arr = np.array(im.resize((IMG_WIDTH, IMG_HEIGHT)))
        mask_arr = np.array(mask.resize((IMG_WIDTH, IMG_HEIGHT)))

        val_images[i] = im_arr
        val_masks[i] = mask_arr

   # mean = np.mean(val_images)
   # std = np.std(val_images)
    val_images -= np.mean(val_images)
    val_images /= np.std(val_images)
    val_masks /= 255.0
        
    return np.expand_dims(val_images, axis=3), np.expand_dims(val_masks, axis=3)

def get_data_generators(images, masks, batch_size=32):
    image_datagen = ImageDataGenerator(featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)

    mask_datagen = ImageDataGenerator(featurewise_center=False,
        featurewise_std_normalization=False,
        rescale=1/255.0,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)

    seed = 1
    image_datagen.fit(images, augment=False, seed=seed)

    image_generator = image_datagen.flow(images, seed=seed)
    mask_generator = mask_datagen.flow(masks, seed=seed)
   # image_generator = image_datagen.flow_from_directory(
    #    path+'/train',
     #   target_size=(128,128),
      #  color_mode='grayscale',
       # class_mode=None,
      #  batch_size=batch_size,
      #  seed=seed)

   # mask_generator = mask_datagen.flow_from_directory(
   #     path+'/masks',
   #     target_size=(128,128),
   #     color_mode='grayscale',
   #     class_mode=None,
   #     batch_size=batch_size,
   #     seed=seed)

    #image_val_gen = image_datagen.flow_from_directory(
    #    path+'/train',
    #    target_size=(128,128),
    #    color_mode='grayscale',
    #    class_mode=None,
    #    seed=seed, subset='validation') 

    #mask_val_gen = mask_datagen.flow_from_directory(
    #    path+'/mask',
    #    target_size=(128,128),
    #    color_mode='grayscale',
    #    class_mode=None,
    #    seed=seed, subset='validation') 

    # combine generators into one which yields image and masks
    train_generator = combine_generator(image_generator, mask_generator)
    #validation_generator = combine_generator(image_val_gen, mask_val_gen)

    return train_generator#, validation_generator
    
def train(image_names, learning_rate, epochs, batch_size):
    train_images, train_masks = get_train_data(image_names)
    val_data = get_val_data()
    print(train_images.shape)
    print(len(val_data[0]))

#    train_gen = get_data_generators('ultrasound-nerve-segmentation', train_images)
    train_gen = get_data_generators(train_images, train_masks)
    model = build_net(IMG_WIDTH, IMG_HEIGHT, batch_size, learning_rate)
    checkpoint = ModelCheckpoint('model_weights_augment.hd5', monitor='val_loss')
    history = LossHistory()
    model.fit_generator(generator=train_gen, epochs=epochs, steps_per_epoch=int(math.ceil(len(train_images)*0.8*18/batch_size)), 
            verbose=1, validation_data=val_data, 
        callbacks=[checkpoint, history]) #WandbCallback()

    return model

def predict_and_score(image_names):
    # get dice coeff
    print("loading model and predicting")
    X, _ = get_train_data(image_names)
    model = load_model('model_weights_augment.hd5', compile=False)
    img_masks = model.predict(X, batch_size=256, verbose=1)
    # resize
    masks_resized = np.zeros((img_masks.shape[0], ORIGINAL_HEIGHT, ORIGINAL_WIDTH))
    for i in range(img_masks.shape[0]):
        im_resized = np.array(cv2.resize(img_masks[i], (ORIGINAL_WIDTH, ORIGINAL_HEIGHT)))
        masks_resized[i] = im_resized
    print("Loading ground truth and dicing")
    # load ground truth masks for these images
    labels = np.zeros((len(X), ORIGINAL_HEIGHT, ORIGINAL_WIDTH))
    for f in image_names:
            im = Image.open("ultrasound-nerve-segmentation/masks/masks/"+f+ "_mask.tif")
            labels[i] = np.array(im)

    # get dice between ground truth and predicted
    d = dice(labels, masks_resized)
    import tensorflow as tf
    print(tf.Session().run(d))

def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next())

def main():
    #wandb.init(project="nerve-segmentation")
    train_files = np.array(os.listdir("ultrasound-nerve-segmentation/train/train"))
    splitfile = np.vectorize(lambda x: os.path.splitext(x)[0])  
    train_names = splitfile(train_files)

    model = train(train_names, 1e-5, 50, 32)
    model.save(os.path.join(wandb.run.dir, "model_augment.h5"))

    # predict_and_score(image_names)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_losses =[]
        self.val_losses = []
        self.dices = []
        self.val_dices = []
        open('metrics.json', 'w').close()

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(str(logs.get('loss')))
        self.val_losses.append(str(logs.get('val_loss')))
        self.dices.append(str(logs.get("dice")))
        self.val_dices.append(str(logs.get("val_dice")))
        with open('metrics.json', 'w') as outfile:
            json.dump({
                "loss":self.train_losses, 
                "val_loss":self.val_losses,
                "dice": self.dices,
                "val_dice":self.val_dices}, outfile)


if __name__ == '__main__': main()
