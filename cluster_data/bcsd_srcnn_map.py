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
batch_size=200 # Size of each batch
n_epochs=2000
use_elevation=1
patience=50
if data_size== -1:
    ds='full'
else:
    ds=data_size


# In[4]:


name='divya_bcsd_ds_{data_size}_elevation_{use_elevation}_e_{n_epochs}'.format(data_size=ds,use_elevation=use_elevation,n_epochs=n_epochs)
name


# In[5]:


x1=xr.open_dataset('/home/madhumadi/sd/nc_files/ak_ma_divya_bcsd_cesm.nc')['observation']
y1=xr.open_dataset('/home/madhumadi/sd/nc_files/obs_1920_to_2005_no_leap.nc')['precipitation (mm)']


x=(x1.dropna(dim='lat',how='all')).dropna(dim='lon',how='all')
y=(y1.dropna(dim='lat',how='all')).dropna(dim='lon',how='all')

# In[6]:
train_images=np.asarray(x.fillna(0))
train_labels=np.asarray(y.fillna(0))

mask=xr.where(y[0]>=0,1,np.nan)
# In[7]:


np.random.seed(seed=0)


# In[8]:


import xarray as xr
a='/home/madhumadi/sd/nc_files/india_masked_ele_interp.nc'
elevation=xr.open_dataset(a)
elevation['Band1'].plot()
#felev_temp=np.asarray(elevation['Band1']).reshape([117,118])


# In[46]:


elevation_masked=(elevation['Band1'].fillna(0))*mask


# In[47]:


elevation_masked.plot()


# In[48]:


elevation1=elevation_masked.dropna(dim='lat',how='all')
elevation=elevation1.dropna(dim='lon',how='all')
felev_temp=np.asarray(elevation).reshape([117,118])


# In[49]:



from numpy import *
def fill_na(x, fillval=0):
    where_are_NaNs = isnan(x)
    x[where_are_NaNs] = fillval
    return x
felev_te=fill_na(felev_temp)


# In[50]:


#felev.shape


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


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


# In[11]:


from keras import optimizers
from keras import initializers
sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model = Sequential()
if use_elevation==1:
    model.add(Conv2D(64, kernel_size=9, input_shape=(117,118,2),padding='same',name='con2d_layer1_with_elevation',kernel_initializer=initializers.random_normal(stddev=0.01)))
else:
    model.add(Conv2D(64, kernel_size=9, input_shape=(117,118,1),padding='same',name='con2d_layer1_without_elevation',kernel_initializer=initializers.random_normal(stddev=0.01)))
model.add(BatchNormalization(name='batch_norm_layer1'))
model.add(Activation("relu"))
model.add(Conv2D(32, kernel_size=1,padding='same',kernel_initializer=initializers.random_normal(stddev=0.001)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(1, kernel_size=5,padding='same',kernel_initializer=initializers.random_normal(stddev=0.001)))
model.compile(loss=root_mean_squared_error, optimizer='adam')


# In[12]:


model.summary()


# In[13]:


terNAN=keras.callbacks.TerminateOnNaN()
tensorboard = TensorBoard(log_dir='./{}/logs{}'.format(name,name), histogram_freq=0,write_graph=True, write_images=False)
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau,CSVLogger
csv_logger = CSVLogger('./{}/training{}.log'.format(name,name))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9,patience=patience, min_lr=0.0001,verbose=1)
mc = ModelCheckpoint('./{}/best_model.h5'.format(name,name), monitor='loss', mode='auto', verbose=0, save_best_only=False)
if use_elevation==1:
    history=model.fit(resu, train_labels.reshape([-1,117,118,1]),batch_size=batch_size, epochs= n_epochs, validation_split = 0.2, verbose=1,callbacks=[es,csv_logger,tensorboard,reduce_lr, mc,terNAN])
else:
    history=model.fit(a.reshape([-1,117,118,1]), b.reshape([-1,117,118,1]),batch_size=batch_size, epochs= n_epochs, validation_split = 0.2, verbose=1,callbacks=[es,csv_logger,tensorboard,reduce_lr, mc,terNAN])


# In[14]:




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







