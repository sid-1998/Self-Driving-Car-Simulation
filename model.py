
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from Dataset_generation import generate_data, Input_shape


# In[2]:


data_X, data_Y = generate_data()


# In[3]:


X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size = 0.2)


# In[4]:


print(X_train.shape, Y_train.shape)
print(X_test.shape,Y_test.shape, )


# In[5]:


model = Sequential()
model.add(Conv2D(24, 5, 5, activation='relu',input_shape = INPUT_SHAPE, subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
# In[6]:


checkpoint = ModelCheckpoint('model-{val_loss:.4f}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')


# In[7]:


model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])


# In[8]:


model.fit(X_train, Y_train, batch_size=32, epochs=30, validation_data=[X_test, Y_test], callbacks=[checkpoint], shuffle=True)
