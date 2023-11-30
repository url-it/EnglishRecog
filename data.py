# This file contains the accuracy and loss of the model for each epoch, and other forms
import pandas as pd
import matplotlib.pyplot as plt

filePath= 'results.csv'

# Shows the accuracy and loss of the model for each epoch

df = pd.read_csv(filePath)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(df['accuracy'], df['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Validation Accuracy')

plt.subplot(1,2,2)
plt.scatter(df['loss'], df['val_loss'])
plt.title('Loss')
plt.xlabel('Loss')
plt.ylabel('Validation Loss')
plt.show()
