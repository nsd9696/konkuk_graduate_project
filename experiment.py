from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img,img_to_array

image = load_img("./static/raw_image/2.png", color_mode='grayscale')
image = img_to_array(image).astype('float32')/255.

image = np.asarray(image)

image = np.expand_dims(image,axis=2)
image = np.expand_dims(image,axis=0)

#예측
model=keras.models.load_model('./model/denoise_sample.h5')
prediction = model.predict(image)
# result = np.squeeze(prediction.astype('uint8'))
result = np.squeeze(prediction)
result_image = Image.fromarray((result*255).astype('uint8'), 'L')
print(result_image)

save_path = './static/denoised_image'

result_image.save(f"{save_path}/sample.png")