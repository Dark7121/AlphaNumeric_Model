import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mnist import MNIST
#mnist is a python library which has the EMNIST and MNIST datasets. 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from emnist import extract_training_samples
from emnist import list_datasets
from emnist import extract_test_samples
list_datasets()
train_images, train_labels = extract_training_samples('byclass')
print(train_images.shape)
print(train_labels.shape)

test_images, test_labels = extract_test_samples('byclass')
print(test_images.shape)
print(test_labels.shape)

train_images = tf.keras.utils.normalize(train_images, axis = 1)
test_images = tf.keras.utils.normalize(test_images, axis = 1)

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)
# specify the arguments
rotation_range_val = 15
width_shift_val = 0.10
height_shift_val = 0.10
# create the class object
train_datagen = ImageDataGenerator(rotation_range = rotation_range_val,
                             width_shift_range = width_shift_val,
                             height_shift_range = height_shift_val)
# fit the generator
train_datagen.fit(train_images.reshape(train_images.shape[0], 28, 28, 1))

# define number of rows & columns
num_row = 4
num_col = 8
num= num_row*num_col
# plot before
print('BEFORE:\n')
# plot images
fig1, axes1 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
     ax = axes1[i//num_col, i%num_col]
     ax.imshow(train_images[i], cmap='gray_r')
     ax.set_title('Label: {}'.format(train_labels[i]))
plt.tight_layout()
plt.show()
# plot after
print('AFTER:\n')
fig2, axes2 = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for X, Y in train_datagen.flow(train_images.reshape(train_images.shape[0], 28, 28, 1),train_labels.reshape(train_labels.shape[0], 1),batch_size=num,shuffle=False):
     for i in range(0, num):
          ax = axes2[i//num_col, i%num_col]
          ax.imshow(X[i].reshape(28,28), cmap='gray_r')
          ax.set_title('Label: {}'.format(int(Y[i])))
     break
plt.tight_layout()
plt.show()
val_datagen = ImageDataGenerator()
val_datagen.fit(test_images.reshape(test_images.shape[0], 28, 28, 1))
val_datagen

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(62, activation = 'softmax')
])
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_datagen.flow(train_images, train_labels, batch_size=128), validation_data= val_datagen.flow(test_images, test_labels, batch_size=64), epochs= 20)

scores = model.evaluate(test_images,test_labels)
print("Accuracy: %.2f%%"%(scores[1]*100))

model.save("my_model.keras")
model.save_weights("my_model.h5")
