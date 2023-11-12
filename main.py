import numpy as numpy
import pandas as pd
from emnist import extract_training_samples
from emnist import extract_test_samples
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
plt.figure()
plt.imshow(testImgs[3])
plt.colorbar()
plt.grid(False)
plt.show()

