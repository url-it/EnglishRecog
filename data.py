# This file contains the accuracy and loss of the model for each epoch, and other forms
import pandas as pd
import matplotlib.pyplot as plt

filePath= 'results.csv'

df = pd.read_csv(filePath)
# print(df)

df['ratio'] = df['accuracy']/df['val_accuracy']
df['lossRatio'] = df['loss']/df['val_loss']
df['ratio'] = df['ratio'].round(4)
df['lossRatio'] = df['lossRatio'].round(4)

mf = df[['ratio', 'lossRatio']]
mf = mf.mean()
mf = mf.round(4)
print(mf)

# run only once 
# df.to_csv('ratios.csv', index=False)

totalAcc = df['accuracy'].sum()
totalValAcc = df['val_accuracy'].sum()
totalLoss = df['loss'].sum()
totalValLoss = df['val_loss'].sum()

ratio = totalAcc/totalValAcc
lossRatio = totalLoss/totalValLoss

# Shows the ratio of the accuracy and loss of the model for each epoch 

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)    
plt.plot(df['ratio'])
plt.title('Accuracy Ratio')
plt.xlabel('Epoch')
plt.ylabel('Accuracy Ratio')

plt.subplot(1,2,2)
plt.plot(df['lossRatio'])
plt.title('Loss Ratio')
plt.xlabel('Epoch')
plt.ylabel('Loss Ratio')
plt.show()

# Shows the accuracy and loss of the model for each epoch

# df = pd.read_csv(filePath)
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.scatter(df['accuracy'], df['val_accuracy'])
# plt.title('Accuracy')
# plt.xlabel('Accuracy')
# plt.ylabel('Validation Accuracy')

# plt.subplot(1,2,2)
# plt.scatter(df['loss'], df['val_loss'])
# plt.title('Loss')
# plt.xlabel('Loss')
# plt.ylabel('Validation Loss')
# plt.show()
