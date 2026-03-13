from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from sklearn.cluster import KMeans
import uvicorn
import ast

app = FastAPI(title="API Bordado")

# Permite que o Netlify converse com o Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def apply_stitch_effect(mask, color_rgb):
    h, w = mask.shape
    stitches = np.zeros((h, w, 3), dtype=np.uint8)
    stitches[mask == 255] = color_rgb
    
    # Textura básica de linhas
    y, x = np.mgrid[0:h, 0:w]
    lines_pattern = (x + y) % 4 == 0 
    dark_color = [max(0, c - 50) for c in color_rgb]
    stitches[(mask == 255) & lines_pattern] = dark_color

    # Efeito 3D falso (Sobel)
    gray_stitches = cv2.cvtColor(stitches, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray_stitches, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_stitches, cv2.CV_64F, 0, 1, ksize=3)
    
    emboss = (sobel_x + sobel_y) / 2.0
    emboss = np.clip(emboss, -50, 50).astype(np.int8)

    result = stitches.astype(np.int16)
    for i in range(3):
        result[:,:,i] += emboss
        
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.bitwise_and(result, result, mask=mask)

@app.post("/gerar_paleta/")
async def gerar_paleta(file: UploadFile = File(...), num_cores: int = Form(5)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_small = cv2.resize(img, (100, 100))
    pixels = img_small.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=num_cores, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    
    return {"paleta": [rgb_to_hex(color) for color in colors]}

@app.post("/aplicar_bordado/")
async def aplicar_bordado(file: UploadFile = File(...), cores_selecionadas: str = Form(...)):
    cores_hex = ast.literal_eval(cores_selecionadas)
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img_rgb.shape
    imagem_final = np.full((h, w, 3), (230, 220, 200), dtype=np.uint8) # Fundo Bege
    
    for hex_color in cores_hex:
        rgb_target = np.array(hex_to_rgb(hex_color))
        lower_bound = np.clip(rgb_target - 30, 0, 255)
        upper_bound = np.clip(rgb_target + 30, 0, 255)
        
        mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        
        borda_processada = apply_stitch_effect(mask, rgb_target)
        idx = mask == 255
        imagem_final[idx] = borda_processada[idx]

    imagem_final_bgr = cv2.cvtColor(imagem_final, cv2.COLOR_RGB2BGR)
    _, encoded_img = cv2.imencode('.png', imagem_final_bgr)
    return Response(content=encoded_img.tobytes(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)