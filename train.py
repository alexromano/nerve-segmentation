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
from tensorflow.keras.callbacks import Callback
import math
import cv2

ORIGINAL_WIDTH = 420
ORIGINAL_HEIGHT = 580
IMG_WIDTH = 128
IMG_HEIGHT = 128

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
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(2)(conv1)

    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(2)(conv2)

    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(conv3)
    # conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(2)(conv3)

    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(pool3)
    # conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(conv4)
    # conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(2)(conv4)

    conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(pool4)
    # conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(conv5)
    # conv5 = BatchNormalization()(conv5)

    conv6 = concatenate([Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv5), conv4])
    # conv6 = Dropout(rate=0.5)(conv6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)

    conv7 = concatenate([Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv6), conv3])
    # conv7 = Dropout(rate=0.5)(conv7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)

    conv8 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv7), conv2])
    # conv8 = Dropout(rate=0.5)(conv8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)

    conv9 = concatenate([Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv8), conv1])
    # conv9 = Dropout(rate=0.5)(conv9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
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

def get_batches(data, batch_size, train_mean, train_std, train=True):
    while True:
        for i in range(0, len(data), batch_size):
            images = []
            labels = []
            for img_name in data[i:i+batch_size]:
                im = Image.open("ultrasound-nerve-segmentation/train/train/"+img_name+".tif")
                lb = Image.open("ultrasound-nerve-segmentation/masks/masks/"+img_name+"_mask.tif")
                im_arr = preprocess(im)
                lb_arr = preprocess(lb)
                images.append(im_arr)
                labels.append(lb_arr)
                
                if train: #and np.any(lb > 0.5):
                    aug_ims, aug_lbs = augment(im_arr, lb_arr)
                    for im in aug_ims:
                        images.append(im)
                    for lb in aug_lbs:
                        labels.append(lb)
            ims = np.expand_dims(np.array(images, dtype='float32'), axis=3)
            lbls = np.expand_dims(np.array(labels, dtype='float32'), axis=3)

            ims -= train_mean
            ims /= train_std
            lbls /= 255.0

            yield ims, lbls
    
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
        # aug_ims, aug_masks = augment(im_arr, mask_arr)

        # j = i * 6 
        train_images[i] = im_arr
        train_masks[i] = mask_arr
        # train_images[j+1:j+6] = aug_ims
        # train_masks[j+1:j+6] = aug_masks
        
    
    mean = np.mean(train_images)
    std = np.std(train_images)
    train_images -= mean
    train_images /= std
    train_masks /= 255.0 
    return np.expand_dims(train_images, axis=3), np.expand_dims(train_masks, axis=3)
    
def train(X, Y, learning_rate, epochs, batch_size):
    model = build_net(IMG_WIDTH, IMG_HEIGHT, batch_size, learning_rate)
    checkpoint = ModelCheckpoint('model_weights_overfit.hd5', monitor='val_loss')
    history = LossHistory()
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_split=0.2, callbacks=[checkpoint, history])
    return model

def predict_and_score(image_names):
    # get dice coeff
    print("loading model and predicting")
    X, _ = get_train_data(image_names)
    # load model and predict on images
    model = load_model('model_weights_overfit.hd5', compile=False)
    img_masks = model.predict(X, batch_size=3, verbose=1)
    # resize
    masks_resized = np.zeros((img_masks.shape[0], ORIGINAL_HEIGHT, ORIGINAL_WIDTH))
    for i in range(img_masks.shape[0]):
        im_resized = np.array(cv2.resize(img_masks[i], (ORIGINAL_WIDTH, ORIGINAL_HEIGHT)))
        masks_resized[i] = im_resized
    print("Loading ground truth and dicing")
    # load ground truth masks for these images
    labels = np.zeros((len(X), ORIGINAL_WIDTH, ORIGINAL_HEIGHT))
    for f in image_names:
            im = Image.open("ultrasound-nerve-segmentation/masks/masks/"+f+ "_mask.tif")
            labels[i] = np.array(im)

    # get dice between ground truth and predicted
    d = dice(labels, masks_resized)
    import tensorflow as tf
    print(tf.Session().run(d))

def main():
    files = np.array(os.listdir("ultrasound-nerve-segmentation/train/train"))
    # image_names = files[np.where(np.char.find(files, '_mask')<0)]
    splitfile = np.vectorize(lambda x: os.path.splitext(x)[0])
    image_names = splitfile(files)

    train_images, train_masks = get_train_data(image_names)

    model = train(train_images, train_masks, 1e-5, 20, 32)

    # predict_and_score(image_names)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.val_losses = []
        self.dices = []
        self.val_dices = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.dices.append(logs.get("dice"))
        self.val_dices.append(logs.get("val_dice"))

if __name__ == '__main__': main()