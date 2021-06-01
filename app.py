import uvicorn
from fastapi import FastAPI,Request,Response,File,UploadFile,Form
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ml.quadtree_image import QuadTree
import shutil
import os
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow import keras

#create app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

#index.route
@app.get('/', response_class=HTMLResponse)
async def index(request:Request):
    return templates.TemplateResponse("index.html",{'request':request})

@app.post("/predict")
async def compress(request:Request, compressed_path: str=Form(...)):
    print(compressed_path)
    image = Image.open('.'+compressed_path)
    print(np.array(image).shape)

    #데이터 전처리
    image = image.resize((63,63))
    image = np.asarray(image)
    if image.dtype == np.uint8:
        image = image/255.0
    image = np.expand_dims(image,axis=0)
    image = np.asarray(image).astype('float32')

    #예측
    model=keras.models.load_model('./model/Beans.h5')
    prediction = model.predict(image)

    label = np.squeeze(prediction)
    np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
    #print(label)

    return templates.TemplateResponse("index_result.html",{'request':request,'label':label,'compressed_path':compressed_path})



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)