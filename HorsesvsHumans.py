#Modules
import os
import cv2
import numpy as np
from keras import models as m
from keras import layers as l
from keras import optimizers as o
from keras import losses as ls
import matplotlib.pyplot as plt

#The dimension of the data is defined
width=height=128

#This function will load the data of the dataset
def load_data(a):
    data = []
    labels = []
    classes = ['Horses', 'Humans']
    for cl in range(len(classes)):
        dir = os.path.join(a, classes[cl])

        for name in os.listdir(dir):
            path = os.path.join(dir, name)
            img = cv2.imread(path)
            img = cv2.resize(img, (width, height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data.append(img)
            labels.append(cl)

    return np.array(data), np.array(labels)

#The function is call to load the train and the validation data
X_train,y_train=load_data(os.getcwd()+'\\Train')
X_val,y_val=load_data(os.getcwd()+'\\Val')

#Building the model
model=m.Sequential([
    l.InputLayer(shape=(width,height,1)),
    l.Conv2D(filters=4,kernel_size=3,padding='valid',activation='relu'),
    l.BatchNormalization(),
    l.MaxPool2D(pool_size=2,strides=2),
    l.Dropout(rate=0.75),
    l.Flatten(),
    l.Dense(3,activation='relu'),
    l.BatchNormalization(),
    l.Dense(1,activation='sigmoid'),

])
print(model.summary())

#Compile the model
model.compile(optimizer=o.Adam(learning_rate=0.01),loss=ls.BinaryCrossentropy,metrics=['accuracy'])

#Training the model
epochs=5
history=model.fit(X_train,y_train,batch_size=32,epochs=epochs,validation_data=(X_val,y_val))

#Two graphs that show what happen in the training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy','val_accuracy'])
plt.show()

#Saving the model with a val_accuracy=0.9023 and a val_loss=0.6454
model.save('datasethoursesvshumans.h5')
