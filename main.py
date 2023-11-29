import numpy as np
import pandas as pd
from emnist import extract_training_samples
from emnist import extract_test_samples
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dropout, Dense, MaxPool2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
# from tensorflow.keraas.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
# from tensorflow.keras.models import Model

# On this one you guys might need to download it using pip
# In order for this to work

# <-- Run this to make sure you guys have the dataset on hand -->
# from emnist import list_datasets
# print(list_datasets())
# <--- comment this out after you run it once --->

# This is the actual datasets split into training and testing
trainImgs, trainLbls = extract_training_samples('letters')
testImgs, testLbls = extract_test_samples('letters')

# Run this to see the image of what we are working with
# plt.figure()
# plt.imshow(testImgs[1000])
# plt.colorbar()
# plt.grid(False)
# plt.show()

trainImgs = np.expand_dims(trainImgs, -1)
testImgs = np.expand_dims(testImgs, -1)
# print(trainImgs.shape)

# Actual Model
# Setting the number of specific labels for the neural network
numOfLbls = len(set(trainLbls))

# Creating the convolution layers for the neural network
inputLayer = Input(shape = trainImgs[0].shape)
conv = Conv2D(32, (3,3), strides = 2, activation= 'relu')(inputLayer)
maxPool = MaxPool2D((2, 2))(conv)

conv2 = Conv2D(64, (3,3), strides = 2, activation= 'relu')(maxPool)

flatCon = Flatten()(conv2)
dropOne = Dropout(0.2)(flatCon)

dense = Dense(512, activation = 'relu')(dropOne)

dropTwo = Dropout(0.2)(dense)

outPutLayer = Dense(numOfLbls+1, activation='softmax')(dropTwo)


model = Model(inputLayer, outPutLayer)
model.compile(optimizer = 'adam', loss ='sparse_categorical_crossentropy', metrics =['accuracy'])

modelHistory = model.fit(trainImgs, trainLbls,
batch_size = 64, epochs=20, validation_data=(testImgs, testLbls), verbose=1)

# <-- End of Model -->