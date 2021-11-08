#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


# In[2]:


import numpy as np
import xarray as xr
import pylab as plt


# In[3]:


data_size=-1 #train+test
batch_size=100 # Size of each batch
n_epochs=1000
use_elevation=1
patience=100
if data_size== -1:
    ds='full'
else:
    ds=data_size


# In[4]:


name='srcnn_tj_1950_1980_f_ds_{data_size}_elevation_{use_elevation}_e_{n_epochs}'.format(data_size=ds,use_elevation=use_elevation,n_epochs=n_epochs)
name


# In[5]:


y=xr.open_dataset('/home/madhumadi/sd/s_data/nc_files/obs_1920_to_2005_no_leap.nc')['precipitation (mm)'].fillna(0)


# In[6]:


xa=y[0:data_size,0:128,0:134].interp(lon=np.arange(66.5,100.0,0.5), lat=np.arange(6.5,38.5,0.5)).fillna(0)
x=xa[0:data_size].interp(lon=np.arange(66.5,100.0,0.25), lat=np.arange(6.5,38.5,0.25)).fillna(0)
test_images=np.asarray(x.sel(time = slice('1981-01-01','2005-12-31')))
train_images=np.asarray(x.sel(time = slice('1950-01-01','1980-12-31')))
test_labels=np.asarray(y[0:data_size,0:128,0:134].sel(time = slice('1981-01-01','2005-12-31')))
train_labels=np.asarray(y[0:data_size,0:128,0:134].sel(time = slice('1950-01-01','1980-12-31')))


# xa=y[0:data_size,0:128,0:134].interp(lon=np.arange(66.5,100.0,0.5), lat=np.arange(6.5,38.5,0.5)).fillna(-10000)
# x=xa[0:data_size].interp(lon=np.arange(66.5,100.0,0.25), lat=np.arange(6.5,38.5,0.25)).fillna(-10000)
# test_images=np.asarray(x.sel(time = slice('2000-01-01','2005-12-31')))
# train_images=np.asarray(x.sel(time = slice('1990-01-01','2000-01-01')))
# test_labels=np.asarray(y[0:data_size,0:128,0:134].sel(time = slice('2000-01-01','2005-12-31')))
# train_labels=np.asarray(y[0:data_size,0:128,0:134].sel(time = slice('1990-01-01','2000-01-01')))

# xa=y[0:data_size,0:128,0:134].interp(lon=np.arange(66.5,100.0,0.5), lat=np.arange(6.5,38.5,0.5)).fillna(0)
# x=xa[0:data_size].interp(lon=np.arange(66.5,100.0,0.25), lat=np.arange(6.5,38.5,0.25)).fillna(0)
# train_images=np.asarray(x)
# train_labels=np.asarray(y[0:data_size,0:128,0:134])

# In[7]:


xa[0].plot()


# In[8]:


if use_elevation==1:
    import xarray as xr
    from numpy import *
    a='/home/madhumadi/sd/s_data/nc_files/india_masked_ele_interp.nc'
    elevation=xr.open_dataset(a)
    elevation_temp1=elevation.interp(lon=np.arange(66.5,100.0,0.5), lat=np.arange(6.5,38.5,0.5))
    elevation_temp2=elevation.interp(lon=np.arange(66.5,100.0,0.25), lat=np.arange(6.5,38.5,0.25))

    felev_temp=np.asarray(elevation_temp2['Band1']).reshape([128,134])

    def fill_na(x, fillval=0):
        where_are_NaNs = isnan(x)
        x[where_are_NaNs] = fillval
        return x
    felev_te=fill_na(felev_temp)
    felev=((felev_te-np.min(felev_te))/(np.max(felev_te)-np.min(felev_te)))

    train_images.shape
    p1=[]
    for i in range(len(train_images)):
        p2=np.asarray([train_images[i],felev*50])
        p1.append(p2)
    p3=np.asarray(p1)
    resu=p3.transpose(0,2,3,1)


# In[9]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation
from keras.layers.normalization import BatchNormalization
import numpy as np
import pickle
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import h5py
import matplotlib.pyplot as plt
from keras import backend as K
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


# In[10]:


"""
def root_mean_squared_error(y_true, y_pred):
        print(y_pred.shape)
        diff=[]
        for i in range(batch_size):
            for j in range(128):
                for k in range(134):
                    if y_pred[i][j][k]==0:
                        continue
                    else:
                        diff1=y_true[i][j][k]-y_pred[i][j][k]
                        diff.append(diff1)
        return K.sqrt(K.mean(K.square(diff)))
"""


# In[55]:



# In[11]:


"""def root_mean_squared_error(y_true, y_pred):
    nonzero = K.any(K.not_equal(y_pred, 0.0), axis=-1)
    n = K.sum(K.cast(nonzero, 'float32'), axis=1)
    return K.sqrt(K.square(y_true-y_pred)/n  )
    """


# In[12]:

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

# In[13]:


from keras import optimizers
from keras import initializers
sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model = Sequential()
if use_elevation==1:
    model.add(Conv2D(64, kernel_size=9, input_shape=(128,134,2),padding='same',name='con2d_layer1_with_elevation',kernel_initializer=initializers.random_normal(stddev=0.01)))
else:
    model.add(Conv2D(64, kernel_size=9, input_shape=(128,134,1),padding='same',name='con2d_layer1_without_elevation',kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(BatchNormalization(name='batch_norm_layer1'))
model.add(Activation("relu"))
model.add(Conv2D(32, kernel_size=1,padding='same',kernel_initializer=initializers.random_normal(stddev=0.001)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(1, kernel_size=5,padding='same',kernel_initializer=initializers.random_normal(stddev=0.001)))
model.compile(loss=root_mean_squared_error, optimizer='adam')


# In[14]:


model.summary()


# In[21]:


terNAN=keras.callbacks.TerminateOnNaN()
tensorboard = TensorBoard(log_dir='./{}/logs{}'.format(name,name), histogram_freq=0,write_graph=True, write_images=False)
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau,CSVLogger
csv_logger = CSVLogger('./{}/training{}.log'.format(name,name))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9,patience=patience, min_lr=0.0001,verbose=1)
mc = ModelCheckpoint('./{}/best_model.h5'.format(name,name), monitor='loss', mode='auto', verbose=0, save_best_only=False)
if use_elevation==1:
    history=model.fit(resu, train_labels.reshape([-1,128,134,1]),batch_size=batch_size, epochs= n_epochs, validation_split = 0.2, verbose=1,callbacks=[es,csv_logger,tensorboard,reduce_lr, mc,terNAN])
else:
    history=model.fit(train_images.reshape([-1,128,134,1]), train_labels.reshape([-1,128,134,1]),batch_size=batch_size, epochs= n_epochs, validation_split = 0.2, verbose=1,callbacks=[es,csv_logger,tensorboard,reduce_lr, mc,terNAN])


# In[16]:



model_json = model.to_json()
with open("./{}/srcnn_pred_{}_16_06.json".format(name,name), "w") as json_file:
    json_file.write(model_json)

model_json = model.to_json()
with open("./{}/srcnn_pred_{}_16_06.json".format(name,name), "w") as json_file:
    json_file.write(model_json)

model.save_weights("./{}/srcnn_weights_{}_16_06.h5".format(name,name))
print("Saved model to disk")

def his():
    loss_history=history.history['loss']
    val_loss_history=history.history['val_loss']
    return loss_history,val_loss_history


with open('./{}/srcnn_16_06_history{}.pkl'.format(name,name), 'wb') as fp:
    pickle.dump(his(), fp) 


# In[ ]:
