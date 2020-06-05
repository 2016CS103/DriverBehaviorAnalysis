# -*- coding: utf-8 -*-
"""
@author: Faraz

"""

import tensorflow as tf
from tensorflow.python.keras.models import load_model
import cv2
import numpy as np

model = load_model('my_model.h5')
img = cv2.imread('test/c10/7362-12342-25951.jpg')
img = np.asarray(img).astype(np.float32)
#img = tf.image.decode_jpeg(cv2.imread('test/c0/249-7469-16148.jpg'))
img = cv2.resize(img,(128,128))

img = np.reshape(img,[1,128,128,3])
#img = tf.cast(img, tf.float32)
classes = model.predict_classes(img)

print (classes)



# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="DBA_2.2.0-rc3_.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)






#test for a single image
from PIL import Image
import PIL.ImageOps  
import requests
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#imgUrl = "https://sites.google.com/view/dbams/home"
#img = Image.open(requests.get(imgUrl, stream=True).raw)

img = Image.open('test/c10/7362-12342-25951.jpg')

img.load()
img = img.resize((128, 128), PIL.Image.ANTIALIAS)

plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.imshow(img)

# Normalize to [0, 1]
data = np.asarray( img, dtype="int32" ) / 255.0

# Normalize to [-1, 1]
data2 = (np.asarray( img, dtype="int32" ) - 128.0) / 128.0

# Inference on input data normalized to [0, 1]
inputImg = np.expand_dims(data,0).astype(np.float32)
interpreter = tf.lite.Interpreter(model_path="DBA_2.2.0-rc3_.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], inputImg)

interpreter.invoke()

output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Predicted value for [0, 1] normalization. Label index: {}, confidence: {:2.0f}%"
      .format(np.argmax(output_data), 
              100 * output_data[0][np.argmax(output_data)]))



# Inference on input data normalized to [-1, 1]
inputImg = np.expand_dims(data2,0).astype(np.float32)
interpreter = tf.lite.Interpreter(model_path="DBA_2.2.0-rc3_.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], inputImg)

interpreter.invoke()

output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Predicted value for [0, 1] normalization. Label index: {}, confidence: {:2.0f}%"
      .format(np.argmax(output_data), 
              100 * output_data[0][np.argmax(output_data)]))



