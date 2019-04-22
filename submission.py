import numpy as np
from datetime import datetime as dt
from PIL import Image
import os
import csv
import cv2
from tensorflow.keras.models import Model, load_model
from train import preprocess

original_width = 420
original_height = 580

def get_test_images():
    print("Getting test images..")
    files = os.listdir("ultrasound-nerve-segmentation/test")
    test_imgs = []
    for f in files:
        im = Image.open("ultrasound-nerve-segmentation/test/"+f)
        im_arr = preprocess(im)
        test_imgs.append(im_arr)
    mean = np.mean(test_imgs)
    std = np.std(test_imgs)
    test_imgs -= mean
    test_imgs /= std
    return np.expand_dims(np.array(test_imgs, dtype='float32'), axis=3)

def get_predictions(images):
    print("Loading model..")
    model = load_model('model_weights.hd5', compile=False)
    img_masks = model.predict(images, batch_size=256, verbose=1)
    
    img_masks_sized = np.zeros((img_masks.shape[0], original_height, original_width)) 
    for i in range(img_masks.shape[0]):
        img_masks_sized[i] = cv2.resize(img_masks[i], (original_width, original_height))

    img_masks_sized[img_masks_sized >= 0.5] = 1
    img_masks_sized[img_masks_sized < 0.5] = 0
    print(img_masks_sized)
    # with open(str(dt.now())+'.csv', 'w') as f:
    #     writer = csv.writer(f, delimiter=',')
    #     writer.writerow(["image", "pixels"])
    #     for i, mask in enumerate(img_masks_sized):
    #         writer.writerow([str(i + 1), run_length_enc(mask)])

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
    test_images = get_test_images()
    get_predictions(test_images)

# img, rle
# 1, 1 1 4 10

if __name__ == "__main__": submission()