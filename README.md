# English Handwritten Letter Recognition

## Overview

This project involves implements of convolutional neural network (CNN) using TensorFlow and Keras for recognizing handwritten English letters. The EMNIST dataset is used for training and testing the model. The trained model is saved as "EnglishRecModel.h5", and the training history is saved to a CSV file named "results.csv".

## libraries

Ensure that you have the necessary libraries installed before running the code. You can install them using the following:

```bash
pip install numpy pandas emnist tensorflow matplotlib
```

```python
from emnist import list_datasets
print(list_datasets())
```

After running the above commands once, you can comment them out to avoid unnecessary dataset downloads.

## Dataset

The EMNIST dataset is split into training and testing sets. Images and labels for both sets are extracted using the `extract_training_samples` and `extract_test_samples` functions from the EMNIST library.

To see an example image from the dataset, you can use the following code:

```python
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(testImgs[1000])
plt.colorbar()
plt.grid(False)
plt.show()
```

## Model Architecture

The CNN model consists of convolutional layers, max-pooling layers, dropout layers, and dense layers. The architecture is as follows:

1. Input Layer
2. Conv2D Layer (32 filters, 3x3 kernel, ReLU activation, stride 2)
3. MaxPooling2D Layer (2x2 pool size)
4. Conv2D Layer (64 filters, 3x3 kernel, ReLU activation, stride 2)
5. Flatten Layer
6. Dropout Layer (20% dropout rate)
7. Dense Layer (512 units, ReLU activation)
8. Dropout Layer (20% dropout rate)
9. Output Layer (Number of unique labels + 1, softmax activation)

The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.

## Training

The model is trained on the training set with a batch size of 500 for 100 epochs. The validation data is the testing set. The training history, including accuracy and loss metrics, is plotted and saved to "results.csv".

## Results Visualization

Uncomment the code at the end of the script to visualize the training history using matplotlib. Two subplots show the accuracy and loss over epochs.

```python
# Uncomment the code below to visualize the training history
# ...
# plt.show()
```

## Files Generated

- Trained Model: `EnglishRecModel.h5`
- Training History: `results.csv`

Feel free to adjust hyperparameters, model, or dataset parameters to further optimize the model. Feedback is appericated :)
