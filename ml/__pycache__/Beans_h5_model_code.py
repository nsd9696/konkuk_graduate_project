# 1. Library import
#from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import *
import seaborn as sns
import tensorflow_datasets as tfds

import skimage.measure

X_train_raw, Y_train_raw = tfds.as_numpy(tfds.load(
    'beans',
    split='train',
    batch_size=-1,
    as_supervised=True,
))

X_train = X_train_raw/255.

Y_train = to_categorical(Y_train_raw)

X_train = sax_embedding(X_train,4,8,'mean')

X_train = X_train.astype(np.uint8)

X_train = X_train/4

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(63,63,3)))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))
#model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=8, epochs=12, validation_split=0.3)

#joblib.dump(model,'Beans.pkl')
model.save('./model/Beans.h5')