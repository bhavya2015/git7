from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras_preprocessing.image import ImageDataGenerator

model = Sequential()


def model_layers(model ,filter_qnt , size_filter , pool_size):
    model.add(Convolution2D(filter_qnt, (size_filter,size_filter) , padding='same', input_shape=(64, 64, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size,pool_size), strides=(2,2)))
    
def add_dense_layer(neurons):   
    model.add(Dense(neurons))
    model.add(Activation("relu"))

model_layers(model ,32,3,2 )
#l2model_layers(model ,layer2)
#l3model_layers(model ,layer3)
#l4model_layers(model ,layer4)

#fullyconnected
model.add(Flatten())
add_dense_layer(128)
#d2add_dense_layer(densel2)

model.add(Dense(units=1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=800,
        epochs=8,
        validation_data=test_set,
        validation_steps=80)

#model.save('cat_dog.h5')

score = model.evaluate(test_set)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
accuracy=score[1]*100

file=open('result.txt','w')
file.write(str(accuracy))
file.close()

