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

model = tf.keras.models.load_model('D:/my_model.keras')
model.load_weights("D:/my_model.h5")


user_test = "D:/Github/Handwritten-Digit-And-Alphabet-Recognizer/Output_SS/output2.png"
col = Image.open(user_test)
gray = col.convert('L')
bw = gray.point(lambda x: 0 if x<100 else 255, '1')
bw.save("bw_image.jpg")
bw
img_array = cv2.imread("bw_image.jpg", cv2.IMREAD_GRAYSCALE)
img_array = cv2.bitwise_not(img_array)
print(img_array.size)
plt.imshow(img_array, cmap = plt.cm.binary)
plt.show()
img_size = 28
new_array = cv2.resize(img_array, (img_size,img_size))
plt.imshow(new_array, cmap = plt.cm.binary)
plt.show()
new_array = np.expand_dims(new_array, axis=0)
user_test = tf.keras.utils.normalize(new_array, axis = 1)
characters = ['0','1','2','3','4','5','6','7','8','9',
'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
predicted = model.predict([[user_test]])
a = predicted[0][0]
for i in range(0,62):
  b = predicted[0][i]
  #print("Probability Distribution for",i,b)

print("The Predicted Value is",characters[np.argmax(predicted[0])])