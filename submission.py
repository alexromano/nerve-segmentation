import numpy as np
from datetime import datetime as dt
from PIL import Image
import os
import csv
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model

original_width = 580
original_height = 420
IMG_WIDTH = 128
IMG_HEIGHT = 128
test_dir = "ultrasound-nerve-segmentation/test"

def get_test_images():
    print("Getting test images..")
    files = os.listdir(test_dir)
    num_files = len(files)
    
    test_imgs = np.zeros((num_files, 128, 128), dtype='float32')
    img_ids = np.zeros((num_files, ), dtype='uint32')

    for i in range(len(files)):
        img_id = int(files[i].split('.')[0])
        im = Image.open(test_dir+"/"+files[i])
        im_arr = np.asarray(im.resize((IMG_WIDTH, IMG_HEIGHT)))
        img_ids[i] = img_id
        test_imgs[i] = im_arr

    mean = np.mean(test_imgs)
    std = np.std(test_imgs)
    test_imgs -= mean
    test_imgs /= std
    return np.expand_dims(test_imgs, axis=3), img_ids

def get_predictions(images, ids):
    print("Loading model..")
    model = load_model('model_weights_augment.hd5', compile=False)
    img_masks = model.predict(images, batch_size=256, verbose=1)

    # crucial to sort predictions in order for the kaggle evaluation
    argsort = np.argsort(ids)
    ids = ids[argsort]
    img_masks = img_masks[argsort]

    print("Running predictions")
    img_masks_sized = np.zeros((img_masks.shape[0], original_height, original_width)) 
    for i in range(img_masks.shape[0]):
        img_masks_sized[i] = cv2.resize(img_masks[i], (original_width, original_height))
    np.save("predictions.npy", img_masks_sized)
    print(len(img_masks_sized))
    print("Saving predictions")
    with open(str(dt.now())+'.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["img", "pixels"])
        for i, mask in enumerate(img_masks_sized):
            mask = (mask > 0.5).astype('uint8')
            print(ids[i])
            writer.writerow([str(ids[i]), run_length_enc(mask)])

# run length encoding
def run_length_enc(label):
    from itertools import chain
    label = label.flatten(order='F')
    y = np.where(label > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

def submission():
    test_images, img_ids = get_test_images()
    get_predictions(test_images, img_ids)

# img, rle
# 1, 1 1 4 10

if __name__ == "__main__": submission()
