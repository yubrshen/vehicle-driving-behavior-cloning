#Reading the driving log to match stearing information to Images
import csv
import random
import cv2, os
import numpy as np

from model import ch, row, col  # camera format
from model import new_row, new_col # after cropping

def record_list(data_directories):
    rec_lst = []
    for d in data_directories:
        catelog_file = d + 'driving_log.csv'
    with open(catelog_file, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r['dir'] = d
            rec_lst.append(r)
    return rec_lst

def balance_records(records, threshold, percentage_keep):
    """
    Given the records keep records with steering angle smaller
    than the threshold with some percentage.
    """
    new_records = []
    dropped_records = []
    for r in records:
        if abs(float(r['steering'])) < threshold:
            if percentage_keep < random.uniform(0, 1):  # simulate the probability to drop the record
                dropped_records.append(r)
                continue
        new_records.append(r)
    return new_records, dropped_records

def shuffle_balance_split(data_sources, split=10, steering_threshold=0.03,
                          steering_keep_percentage=0.1, sequential=False):
    """
    Prepare sample records and split them into training and validation set,
    (One split-th would be used as validation, the rest would be training)
    with filtering, and random shuffling, if required. 
    """
    records, dropped = balance_records(record_list(data_sources),
                                       steering_threshold, steering_keep_percentage)
    # changd from 0.3 to 0.1, to 0, to 0.1 threshold from 0.05 to 0.1, to 0.05
    if not sequential:
        records = random.sample(records, len(records)) 
    train_list = [records[i] for i in range(len(records)) if (i % split) != 0]
    validation_list = [records[i] for i in range(len(records)) if (i % split) == 0]
    return train_list, validation_list, dropped

import skimage.transform as sktransform

def preprocess(image, top_offset=.375, bottom_offset=.125): # experimented with top_offset, 0.25, 0.3, no good. 
    """
    Crops an image by `top_offset` and `bottom_offset` portions of image, 
    resizes to 32x128 px and scales pixel values to [0, 1].
    """
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    image = sktransform.resize(image[top:-bottom, :], (new_row, new_col, 3))
    return image

import matplotlib.image as mpimg

def image_and_steering(record, augment):
    """
    Returns image, and associated steering angle,
    by randomly selecting from left, center, and right camera,
    compensate the steering angle, if needed.
    """
    cameras = ['left', 'center', 'right']
    cameras_steering_correction = [.25, 0., -.25]
    # Randomly select camera
    camera_idx = np.random.randint(len(cameras)) if augment else 1
    camera = cameras[camera_idx]
    # Read image and work out steering angle
    image = mpimg.imread(os.path.join(record['dir'],
                                      record[camera].strip()))
    # mping.imread reads image in RGB channel order consistent with that used in drive.py
    # Image.open(BytesIO(base64.b64decode(imgString)))
    angle = float(record['steering']) + cameras_steering_correction[camera_idx]
    return image, angle

def shadow(image):
    # Add random shadow as a vertical slice of image
    h, w = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)  # determine two random position on horizontal
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    return image

def gen_samples(records, batch_size=128, augment=True, sequential=False):
    """
    Keras generator yielding batches of training/validation records.
    Applies records augmentation pipeline if `augment` is True.
    """
    #print('enter gen_samples:')
    while True:
        # Generate random batch of indices
        indices = range(len(records)) if sequential else np.random.permutation(len(records))

        for batch in range(0, len(indices), batch_size):
            batch_indices = indices[batch:(batch + batch_size)]
            # Output arrays
            x = np.empty([0, new_row, new_col, 3], dtype=np.float32)
            y = np.empty([0], dtype=np.float32)
            # Read in and preprocess a batch of images
            for i in batch_indices:
                image, angle = image_and_steering(records[i], augment)
                if augment:
                    image = shadow(image)
                # Randomly shift up and down while pre-processing
                v_delta = .05 if augment else 0
                image = preprocess(
                    image,
                    top_offset=random.uniform(.375 - v_delta, .375 + v_delta),
                    bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta)
                )
                # Append to batch
                x = np.append(x, [image], axis=0)
                y = np.append(y, [angle])
            if not sequential:
                # Randomly flip half of images in the batch
                flip_indices = random.sample(range(x.shape[0]), int(x.shape[0] / 2))
                x[flip_indices] = x[flip_indices, :, ::-1, :]
                y[flip_indices] = -y[flip_indices]
            yield (x, y)
