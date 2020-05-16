#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils
import keras


# In[2]:


(x_train, y_train),(x_test,y_test) =mnist.load_data()


# In[3]:


img_rows=x_train[0].shape[0]
img_cols=x_train[0].shape[1]


# In[4]:


x_train = x_train.reshape(x_train.shape[0],img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)


# In[5]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# In[6]:



# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255


# In[7]:


# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[8]:


num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]


# In[9]:


model=Sequential()


# In[10]:


input_shape = (img_rows, img_cols, 1)


# In[11]:


model.add(Conv2D(20, (5,5) , padding='same', input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(50, (5,5) , padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#fullyconnected
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

#softmax for classifivcation
model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])
print(model.summary())


# In[ ]:


# Training Parameters
batch_size = 128
epochs = 10
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

model.save("digit_lenet.h5")


# In[ ]:


scores = model.evaluate(x_test, y_test, verbose=1)


# In[ ]:


print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[ ]:


accuracy=scores[1]


# In[ ]:




