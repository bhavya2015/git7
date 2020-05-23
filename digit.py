#!/usr/bin/env python
# coding: utf-8

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils
import keras

(x_train, y_train),(x_test,y_test) =mnist.load_data()


# In[4]:


img_rows=x_train[0].shape[0]
img_cols=x_train[0].shape[1]


# In[5]:


x_train = x_train.reshape(x_train.shape[0],img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)


# In[6]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')





# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255


# In[8]:


# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[9]:


num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]



model=Sequential()


# In[11]:


input_shape = (img_rows, img_cols, 1)


# In[12]:


def model_layers(model ,filter_qnt , size_filter , pool_size):
    model.add(Conv2D(filter_qnt, (size_filter,size_filter) , padding='same', input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size,pool_size), strides=(2,2)))
    
def add_dense_layer(neurons):   
    model.add(Dense(neurons))
    model.add(Activation("relu"))


# In[13]:


model_layers(model ,10,5,2 )
#model_layers(model ,layer1)
#model_layers(model ,layer3)


#fullyconnected
model.add(Flatten())
add_dense_layer(neurons=100)


#softmax for classifivcation
model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])
print(model.summary())


# In[14]:


# Training Parameters
batch_size = 128
epochs = 1
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)

#model.save("digit_lenet.h5")

scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
accuracy=scores[1]*100

file1=open("result.txt",'w')
file.write(str(accuracy))
file1.close()
