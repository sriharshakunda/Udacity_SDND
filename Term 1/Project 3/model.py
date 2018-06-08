# Import neccesary libraries
import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D,Dropout
import sys




#Cretae image data and steering angle data empty list
img_data       = []
measurements = []
#tune  = 0.001
#steering  = [0, tune, -tune]
prob  = 0.5


# 	Load Sample Data from Udacity provided data set
with open('data\\driving_log.csv') as csvfile:
    reader=csv.reader(csvfile,delimiter=',',skipinitialspace=True)

    for line in reader:
        for i in range(3):
            image=line[i]
            actual=cv2.imread(image)
            #convert BGR images to RGB
            converted=cv2.cvtColor(actual,cv2.COLOR_BGR2RGB)
            #crop the image to remove unnecesary boundaries
            cropped=converted[50:137,:,:]
            #resize the cropped image
            new_image=cv2.resize(cropped,(200,66))

            
            #measurement values
            measurement=float(line[3])
 
            
            flipped=cv2.flip(new_image,1)
            img_data.append(flipped)
            measurements.append(measurement*-1)
print('......................................sample data loaded.................................')


# 	Load data from forward driving data set
with open('forward\\driving_log.csv') as csvfile:
    reader=csv.reader(csvfile,delimiter=',',skipinitialspace=True)

    for line in reader:
     
        for i in range(3):
            image= line[i]
            actual=cv2.imread(image)
            #convert BGR images to RGB
            converted=cv2.cvtColor(actual,cv2.COLOR_BGR2RGB)
            #crop the image to remove unnecesary boundaries
            cropped=converted[50:137,:,:]
            #resize the cropped image
            new_image=cv2.resize(cropped,(200,66))
   
            
            #measurement values
            measurement=float(line[3])

            
            flipped=cv2.flip(new_image,1)
            img_data.append(flipped)
            measurements.append(measurement*-1)
print('......................................forward data loaded.................................')


# 	Load data from reverse driving data set
with open('reverse\\driving_log.csv') as csvfile:
    reader=csv.reader(csvfile,delimiter=',',skipinitialspace=True)

    for line in reader:
        #lines.append(line)
        for i in range(3):
            image= line[i]
            actual=cv2.imread(image)
            #convert BGR images to RGB
            converted=cv2.cvtColor(actual,cv2.COLOR_BGR2RGB)
            #crop the image to remove unnecesary boundaries
            cropped=converted[50:137,:,:]
            #resize the cropped image
            new_image=cv2.resize(cropped,(200,66))

            
            #measurement values
            measurement=float(line[3])

            
            flipped=cv2.flip(new_image,1)
            img_data.append(flipped)
            measurements.append(measurement*-1)
print('......................................reverse data loaded.................................')

#Create numpy array to convert the saved image data and steering angle data into arrays
X_train=np.array(img_data)
y_train=np.array(measurements)
#Create Sequentila Model
model=Sequential()
#Normalize the image data
model.add(Lambda(lambda x: (x/255.0) - 0.5,input_shape=(66, 200, 3)))
#Run Nvidia Deep Learning Neural Network Architecture
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dropout(prob))
model.add(Dense(100))
model.add(Dropout(prob))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
print('Compilation Done')
print(model.summary())

#Store the model data in history_object to use it for plotting the validation and training loss results
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 2, verbose = 1)

#Save the model as 'model.h5' in the current folder
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


