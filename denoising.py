import pandas as pandas
import numpy as numpy
import torch 
from google.colab import files
%tensorflow_version 1.x
import tensorflow as tf
import keras 
from keras import optimizers
import glob
from keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input, BatchNormalization
import matplotlib.pyplot as plt
np.random.seed(111)
#### 주소 맞추기
from google.colab import drive
drive.mount('/content/drive')

train = glob.glob('/content/drive/My Drive/denoising_kaggle/train/*.png')
train_cleaned = glob.glob('/content/drive/My Drive/denoising_kaggle/train_cleaned/*.png')
test = glob.glob('/content/drive/My Drive/denoising_kaggle/test/*.png')
sample_submission = pd.read_csv('/content/drive/My Drive/denoising_kaggle/sampleSubmission.csv')

epochs = 40
batch_size = 16

X = [] #train
X_target = [] #train_label

for img in train:
  img = load_img(img, color_mode='grayscale', target_size=(420,540))
  img = img_to_array(img).astype('float32')/255.
  X.append(img)

for img in train_cleaned:
  img = load_img(img, color_mode='grayscale', target_size=(420,540))
  img = img_to_array(img).astype('float32')/255.
  X_target.append(img)

X = np.array(X)
X_target = np.array(X_target)

plt.imshow(np.squeeze(X[0]),cmap='gray')
print(np.squeeze(X[0]).shape)

imgdatagen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range= 0.2,
    zoom_range=0.2,
    horizontal_flip = True,
    fill_mode='nearest'
)
sample_img = np.expand_dims(X[0],axis=0)
sample_augmen = imgdatagen.flow(sample_img,batch_size=1)
fig = plt.figure(figsize=(30,30))
for i in range(10):
  plt.subplot(2,5,i+1)
  batch = sample_augmen.next()
  image = batch[0].astype('float32')
  plt.imshow(np.squeeze(image), cmap='gray')

test_list = []
for img in test:
  img_pixels = load_img(img, color_mode='grayscale')
  img_array = img_to_array(img_pixels)
  w,h = img_pixels.size
  test_list.append(img_array.reshape(1,h,w,1)/255.)

def build_autoencoder():
  input_img = Input(shape=(None,None,1),name='image_input')
  
  #encoder
  x = Conv2D(32, (3,3), activation='relu',kernel_initializer='he_normal', padding='same')(input_img)
  x = BatchNormalization()(x)
  # x = MaxPooling2D((2,2))(x)
  x = Conv2D(64,(3,3),activation='relu', kernel_initializer='he_normal', padding='same')(x)
  x = MaxPooling2D((2,2),padding='same')(x)

  #decoder
  x = Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(x)
  # x = UpSampling2D((2,2))(x)
  x = BatchNormalization()(x)
  x = Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal', padding='same')(x)
  x = UpSampling2D((2,2))(x)
  x = Conv2D(1,(3,3),activation='sigmoid', kernel_initializer='he_normal', padding='same')(x)

  #model
  autoencoder = Model(inputs=input_img, outputs=x)
  return autoencoder

model = build_autoencoder()
model.compile(optimizer=optimizers.Adam(), loss='MSE')

hist = model.fit(X, X_target, epochs=epochs, batch_size=batch_size)

predicted_list = []
for img in test_list:
  # img = np.reshape(img, (1,420,540,1))
  predicted = model.predict(img)[0,:,:,0]
  predicted_list.append(predicted)

import matplotlib.pyplot as plt
_, ax = plt.subplots(1,2, figsize=(12,9.338))
ax[0].imshow(np.squeeze(test_list[0]),cmap='gray')
ax[1].imshow(np.squeeze(predicted_list[0].astype('float32')),cmap='gray')
plt.show()

sample_submission

id_col = sample_submission['id']
predicted_list = np.array(predicted_list)
print(predicted_list[0].shape)

submit_vector = []
for img in predicted_list:
  x,y = img.shape
  for y_ in range(y):
    for x_ in range(x):
      submit_vector.append(img[x_,y_])
print(len(submit_vector))

value_col = pd.Series(submit_vector, name='value')
submission = pd.concat([id_col,value_col],axis=1)
submission.head()