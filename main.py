import numpy as np
import pandas as pd
from emnist import extract_training_samples
from emnist import extract_test_samples
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dropout, Dense, MaxPool2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

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

# Actual Model

numOfLbls = len(set(trainLbls))

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
batch_size = 500, epochs=100, validation_data=(testImgs, testLbls), verbose=1)

model.save('EnlgishRecModel.h5')

# <-- End of Model -->

# Plotting the accuracy and loss

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(modelHistory.history['accuracy'], label='accuracy')
plt.plot(modelHistory.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1,2,2)
plt.plot(modelHistory.history['loss'], label='loss')
plt.plot(modelHistory.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

filePath = 'results.csv'
df = pd.DataFrame(modelHistory.history)
df.to_csv(filePath, index=False)
