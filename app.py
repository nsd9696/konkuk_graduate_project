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
    # upload image
    content = await file.read()
    with open(f"static/raw_image/{file.filename}", "wb") as f:
        f.write(content)
    file_path = f"/static/raw_image/{file.filename}"

    #compression
    size_mult=int(size_mult)
    detail_threshold=int(detail_threshold)
    max_depth=int(max_depth)
    target_depth=int(target_depth)
    image = Image.open(BytesIO(content))
    image = image.resize((image.size[0] * size_mult, image.size[1] * size_mult))
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



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)