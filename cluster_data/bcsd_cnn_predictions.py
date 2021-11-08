import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import numpy as np
#from keras.utils.vis_utils import plot_model
import pylab as plt
import xarray as xr

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


from keras.models import model_from_json
from keras import backend as K
# load json and create model
json_file = open('/home/madhumadi/sd/bcsd_ds_full_elevation_1_e_2000/srcnn_pred_bcsd_ds_full_elevation_1_e_2000_16_06.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("/home/madhumadi/sd/bcsd_ds_full_elevation_1_e_2000/srcnn_weights_bcsd_ds_full_elevation_1_e_2000_16_06.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss=root_mean_squared_error,optimizer='adam')

#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot





data_size=-1 #train+test
batch_size=100 # Size of each batch
n_epochs=2000
use_elevation=1
patience=50
if data_size== -1:
    ds='full'
else:
    ds=data_size


# In[4]:


name='bcsd_ds_{data_size}_elevation_{use_elevation}_e_{n_epochs}'.format(data_size=ds,use_elevation=use_elevation,n_epochs=n_epochs)
name


# In[5]:

#x=xr.open_dataset('cliped_gcm_1920_to_2005.nc')['__xarray_dataarray_variable__'].fillna(0)
data=xr.open_dataset('/home/madhumadi/sd/nc_files/observation_bcsd_op_1920_2005.nc')
y=data['precipitation (mm)'].fillna(0)


# In[6]:
x=data['bias_corrected_gcm'].fillna(0)
train_images=np.asarray(x)
train_labels=np.asarray(y)

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

print('predicting')
predictions=loaded_model.predict(resu)


import numpy as np
print('saving')
np.save('bcsd_srcnn_predticions.npy',predictions)
