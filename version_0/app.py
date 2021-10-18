import uvicorn
from fastapi import FastAPI,Request,Response,File,UploadFile,Form
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img,img_to_array


#create app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

#index.route
@app.get('/', response_class=HTMLResponse)
async def index(request:Request):
    return templates.TemplateResponse("index.html",{'request':request})

@app.post("/predict")
async def compress(request:Request, file:UploadFile = File(...)):

    content = await file.read()
    with open(f"./static/raw_image/{file.filename}", "wb") as f:
        f.write(content)
    file_path = f"./static/raw_image/{file.filename}"

    image = load_img(file_path, color_mode='grayscale')
    image = img_to_array(image).astype('float32')/255.

    image = np.asarray(image)

    image = np.expand_dims(image,axis=2)
    image = np.expand_dims(image,axis=0)

    #예측
    model=keras.models.load_model('./model/denoise_sample.h5')
    prediction = model.predict(image)
    result = np.squeeze(prediction)
    result_image = Image.fromarray((result*255).astype('uint8'), 'L')

    save_path = './static/denoised_image'

    isdir = os.path.isdir(save_path)

    if isdir == True:
        result_image.save(f"{save_path}/{file.filename}")
    else:
        os.makedirs(save_path)
        result_image.save(f"{save_path}/{file.filename}")
    denoised_path = f"/static/denoised_image/{file.filename}"

    return templates.TemplateResponse("index_result.html",{'request':request,'file_path':file_path,'denoised_path':denoised_path})



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)