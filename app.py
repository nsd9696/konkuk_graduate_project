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

@app.post('/compression')
async def compress(request:Request, file:UploadFile = File(...), 
size_mult: str=Form(...), detail_threshold: str=Form(...), max_depth:str=Form(...), target_depth:str=Form(...)):

    """
    타노스 엔진으로 이미지를 압축한다.
    - size_mult: quad tree 알고리즘 전 이미지 resize parameter
    - detail_threshold: 특정 분할되어 있는 영역의 RGB pixel 값들의 분포 오차의 합에 대한 임계값.
        분포 오차의 합이 detail_threshold보다 클 경우 해당 영역을 한번 더 4분할 한다.
        해당 파라미터가 낮을수록 더 세분화하여 압축한다(압축률이 떨어진다).
    - max_depth: 최대로 이미지를 4분할 할 수 있는 횟수
    - target_depth: 4분할 하고자 하는 횟수
    """
    # upload image
    content = await file.read()
    with open(f"static/raw_image/{file.filename}", "wb") as f:
        f.write(content)
    file_path = f"/static/raw_image/{file.filename}"

    #compression
    size_mult=float(size_mult)
    detail_threshold=int(detail_threshold)
    max_depth=int(max_depth)
    target_depth=int(target_depth)
    image = Image.open(BytesIO(content))
    image = image.resize((int(image.size[0] * size_mult), int(image.size[1] * size_mult)))
    quadtree = QuadTree(image,int(detail_threshold),int(max_depth))
    image,temp = quadtree.create_image(int(target_depth),show_lines=False)
    save_path = './static/thanosed_image'
    isdir = os.path.isdir(save_path)

    if isdir == True:
        image.save(f"{save_path}/{file.filename}")
    else:
        os.makedirs(save_path)
        image.save(f"{save_path}/{file.filename}")
    compressed_path = f"/static/thanosed_image/{file.filename}"
    return templates.TemplateResponse("index.html",{'request':request,'file_path':file_path, 'compressed_path':compressed_path,
    'size_mult':size_mult,'detail_threshold':detail_threshold,'max_depth':max_depth,'target_depth':target_depth})

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