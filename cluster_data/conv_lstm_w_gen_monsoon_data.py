#!/usr/bin/env python
# coding: utf-8

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# 
# #import necessary pacakages
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# import numpy as np
# import pickle
# import tensorflow as tf
# from keras.models import model_from_json
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint
# import numpy
# from numpy import *
# import h5py
# import matplotlib.pyplot as plt
# import xarray 
# import xarray as xr

# In[1]:


import os
gpu=1
if gpu==1:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


# In[2]:


import datetime

currentDT = datetime.datetime.now()


# In[3]:


#from keras import backend as k
#k.set_session(k.tf.Session(config=k.tf.ConfigProto(intra_op_parallelism_threads=23, inter_op_parallelism_threads=23)))
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras import backend as K
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#import pylab as plt
import tensorflow as tf


# In[37]:


import xarray as xr
y=xr.open_dataset('/home/madhumadi/sd/nc_files/obs_1920_to_2005_no_leap.nc')['precipitation (mm)']
x=xr.open_dataset('/home/madhumadi/sd/nc_files/cliped_gcm_1920_to_2005.nc')['__xarray_dataarray_variable__']


# In[39]:


monsoons_x=x#.sel(time=x.time.dt.month.isin([5,6,7,8,9]))
monsoons_y=y#.sel(time=y.time.dt.month.isin([5,6,7,8,9]))
monsoons_xa=monsoons_x.dropna(dim='lat',how='all')
ma=monsoons_xa.dropna(dim='lon',how='all')


# In[40]:


monsoons_ya=monsoons_y.dropna(dim='lat',how='all')
la=monsoons_ya.dropna(dim='lon',how='all')


# In[36]:


import pickle
data_start='1920-01-01'
data_end='2005-12-31'
time_steps=15
n_epochs=100
batch_size=1
patience=20
use_elevation=1
load_model=0
load_weights=0
load_model_weights_path='/home/akash/sd/con_lstm_T1_d_1000_t_15_e_100/conv_lstm_con_lstm_T1_d_1000_t_15_e_10008_06_3.h5'
name='con_lstm_Ker_new_103_d_{data_start}_{data_end}_t_{time_steps}_e_{n_epochs}__{date_time}'.format(data_start=data_start,data_end=data_end,time_steps=time_steps,n_epochs=n_epochs,date_time=(datetime.datetime.now()).strftime("%Y-%m-%d %H:%M:%S"))


# In[41]:


m=ma.sel(time = slice(data_start,data_end))
l=la.sel(time = slice(data_start,data_end))


# In[42]:


train_images_t=np.asarray(m.fillna(0))
train_labels_t=np.asarray(l.fillna(0))

mask=xr.where(y[0]>=0,1,np.nan)


# In[43]:


np.random.seed(seed=0)


# In[44]:


name


# data_size=10 #train+test
# time_steps=3   # timesteps to unroll
# n_units=129 # hidden LSTM units
# n_inputs=129*135*2
# n_output=129*135
# #batch_size=100 # Size of each batch
# n_epochs=100
# dropout_rate=0.3
# batch_size= 3

# In[45]:


import xarray as xr
a='/home/madhumadi/sd/nc_files/india_masked_ele_interp.nc'
elevation=xr.open_dataset(a)
elevation['Band1'].plot()
#felev_temp=np.asarray(elevation['Band1']).reshape([129,135])


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
felev=fill_na(felev_temp)


# In[50]:


felev.shape


# In[51]:


#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

xt,xv,yt,yv=train_test_split(train_images_t,train_labels_t,test_size=0.2)


# In[52]:


yt.shape


# In[53]:

valid_images=np.asarray(xv[1:,:,:]).reshape([-1,117,118,1])
valid_labels=np.asarray(yv[0:,:,:]).reshape([-1,117,118,1])

train_images=np.asarray(xt[1:,:,:]).reshape([-1,117,118,1])
train_labels=np.asarray(yt[0:,:,:]).reshape([-1,117,118,1])


# In[54]:




# In[55]:


from keras.preprocessing.sequence import TimeseriesGenerator
from keras.preprocessing.conv_generation_ele import conv_TimeseriesGenerator
import numpy as np
if use_elevation==1:
    generator = conv_TimeseriesGenerator(train_images, train_labels,felev.reshape([117,118,1]),length=time_steps,batch_size=1)
    valid_generator = conv_TimeseriesGenerator(valid_images, valid_labels,felev.reshape([117,118,1]),length=time_steps,batch_size=1)
else:
    generator = TimeseriesGenerator(train_images, train_labels,length=time_steps,batch_size=1)
    valid_generator = TimeseriesGenerator(valid_images, valid_labels,length=time_steps,batch_size=1)


# In[56]:


if load_model==1:
  json_file = open(load_model_path, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  seq = model_from_json(loaded_model_json)
  print("Loaded model from disk")


# In[57]:


def root_mean_squared_error(y_true, y_pred):
        print('######################################')
        print(type(y_pred))
        denomi=K.cast(K.tf.count_nonzero(y_pred),K.tf.float32)
        print(tf.size(denomi))
        print(tf.size((K.square(y_true-y_pred)  )))
        return (K.tf.reduce_sum(K.square(y_true-y_pred))/denomi  )

#def root_mean_squared_error(y_true, y_pred):
#        return K.sqrt(K.mean(K.square(y_pred - y_true)))
# In[58]:


import keras
from keras import optimizers
from keras import initializers
adam=keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


# In[59]:


seq = Sequential()
if use_elevation==1:
  seq.add(ConvLSTM2D(filters=40, kernel_size=(5, 5),input_shape=(None, 117, 118, 2),padding='same',name='input_117_118_2', return_sequences=True))
else:
  seq.add(ConvLSTM2D(filters=40, kernel_size=(5, 5),input_shape=(None, 117, 118, 1),padding='same',name='input_117_118_1', return_sequences=True))
seq.add(BatchNormalization(name='batch_nor_1'))
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True,name='conv_lstm1'))
seq.add(BatchNormalization(name='batch_nor_2'))
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True,name='conv_lstm2'))
seq.add(BatchNormalization(name='batch_nor_3'))
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=False,name='conv_lstm3'))
seq.add(BatchNormalization(name='batch_nor_4'))

seq.add(Conv2D(1, kernel_size=3,padding='same',name='conv2_output'))#,kernel_initializer=initializers.random_normal(stddev=0.001)))
#seq.add(Conv3D(filters=1, kernel_size=(3, 3,3),activation='relu',padding='same', data_format='channels_last'))


if load_weights==1:
  seq.load_weights(load_model_weights_path)
  print("Loaded model weights from disk")

seq.compile(loss='mean_squared_error',optimizer=adam)


# In[60]:


seq.summary()


# model = Sequential()
# model.add(LSTM( n_units, input_shape=( time_steps,  n_inputs),return_sequences=False,dropout=dropout_rate,recurrent_dropout=dropout_rate))
# model.add(Dense( n_output, activation='relu'))
# model.compile(loss='mean_squared_error',optimizer='adam')
# 

# In[61]:


import os
if not os.path.exists(name):
    os.makedirs(name)
model_json = seq.to_json()
with open("./{}/conv_lstm_pred_{}08_06_3.json".format(name,name), "w") as json_file:
    json_file.write(model_json)


# In[62]:


loss_patience=50
reduce_patience=30


# In[ ]:


terNAN=keras.callbacks.TerminateOnNaN()
tensorboard = TensorBoard(log_dir='./{}/logs{}'.format(name,name), histogram_freq=0,write_graph=True, write_images=False)
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau,CSVLogger
csv_logger = CSVLogger('./{}/training{}.log'.format(name,name))
es1 = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=loss_patience)
es2 = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,patience=reduce_patience, min_lr=0.0001,verbose=1)
mc = ModelCheckpoint('./{}/best_model_{}.h5'.format(name,name), monitor='loss', mode='auto', verbose=0, save_best_only=False)
#history=seq.fit_generator(generator, steps_per_epoch=20,epochs= n_epochs,validation_data=valid_generator,verbose=1,callbacks=[es1,es2,csv_logger,tensorboard,reduce_lr, mc ,terNAN])
history=seq.fit_generator(generator,epochs= n_epochs,steps_per_epoch=1000,verbose=1,callbacks=[csv_logger,tensorboard, mc ,terNAN])

# In[45]:


seq.save_weights("./{}/conv_lstm_{}08_06_3.h5".format(name,name))
print("Saved model to disk")


# In[62]:


#x,y=generator[5]
#x.shape
#plt.imshow(x[0,-1,:,:,0])


# In[ ]:

"""
def his():
    loss_history=history.history['loss']
    val_loss_history=history.history['val_loss']
    return loss_history,val_loss_history


# In[ ]:


with open('./{}/conv_lstm_08_06_3_history{}.pkl'.format(name,name), 'wb') as fp:
    pickle.dump(his(), fp) 


# In[ ]:

"""
#pp=seq.predict(X_train)
