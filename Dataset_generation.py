
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# In[4]:


Image_height, Image_width, Image_channels = 66, 200, 3
Input_shape = (Image_height, Image_width, Image_channels)


# In[5]:


def get_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[6]:


def resize_img(img):
    new_img = cv2.resize(img, (Image_width, Image_height))
    return new_img


# In[7]:


def crop_img(img):
    # new_img = img[70:-25, :, :]
    new_img = img[60:-25,:,:]
    return new_img


# In[8]:


def process_img(path):
    img = get_image(path)
    img = crop_img(img)
    img = resize_img(img)
    img = img/255.0
    return img

def process(image):
    img = crop_img(image)
    img = resize_img(img)
    img = img/255.0
    return img
# In[9]:


def choose_img(row, steering):
    ch = np.random.choice(3)
    img = process_img(row[ch])
    steering_angle = float(steering)
    if ch == 1:
        steering_angle += 0.2
    if ch == 2:
        steering_angle -= 0.2

    return img, steering_angle


# In[10]:


def generate_data():
    ds = pd.read_csv('/home/siddharth/Downloads/beta-simulator-linux/train_data/driving_log.csv')
    data = ds.values
    X = data[:,:3]
    Y = data[:,3]

    data_X = np.empty((X.shape[0], Image_height, Image_width, Image_channels))
    data_Y = np.empty((Y.shape[0]))

    for ix in range(X.shape[0]):
        img, steering = choose_img(X[ix], Y[ix])

        data_X[ix] = img
        data_Y[ix] = steering

    return data_X, data_Y
