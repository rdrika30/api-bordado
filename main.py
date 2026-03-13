from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from sklearn.cluster import KMeans
import uvicorn
import ast
from typing import Optional

app = FastAPI(title="API Bordado Pro - Texturas Dinâmicas")

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

def tile_texture(texture, target_h, target_w):
    """Repete a textura (seamless) para cobrir toda a imagem."""
    h, w = texture.shape[:2]
    reps_y = int(np.ceil(target_h / h))
    reps_x = int(np.ceil(target_w / w))
    if len(texture.shape) == 3:
        tiled = np.tile(texture, (reps_y, reps_x, 1))
    else:
        tiled = np.tile(texture, (reps_y, reps_x))
    return tiled[:target_h, :target_w]

def apply_stitch_effect(mask, color_rgb, textura_fio_gray):
    """Aplica o efeito baseando-se na textura enviada."""
    h, w = mask.shape
    
    # 1. Multiplicação de Cor
    tex_float = textura_fio_gray.astype(np.float32) / 255.0
    stitched = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        stitched[:,:,i] = np.clip(color_rgb[i] * tex_float, 0, 255)

    # 2. Efeito Chanfro e Entalhe
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0.4, 1.1, cv2.NORM_MINMAX)
    
    result_float = np.zeros_like(stitched, dtype=np.float32)
    for i in range(3):
        result_float[:,:,i] = stitched[:,:,i] * dist
        
    result_3d = np.clip(result_float, 0, 255).astype(np.uint8)

    # 3. Micro-sombras
    gray = cv2.cvtColor(result_3d, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    emboss = (sobel_x + sobel_y) * 0.3
    emboss = np.clip(emboss, -40, 40).astype(np.int8)

    final_result = result_3d.astype(np.int16)
    for i in range(3):
        final_result[:,:,i] += emboss
        
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)
    
    return cv2.bitwise_and(final_result, final_result, mask=mask)

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
async def aplicar_bordado(
    file: UploadFile = File(...), 
    cores_selecionadas: str = Form(...),
    textura_fio: Optional[UploadFile] = File(None),
    fundo_tecido: Optional[UploadFile] = File(None)
):
    cores_hex = ast.literal_eval(cores_selecionadas)
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    # PROCESSA O FUNDO ENVIADO PELO USUÁRIO (OU GERA UM BEGE PROVISÓRIO)
    if fundo_tecido and fundo_tecido.filename:
        conteudo_fundo = await fundo_tecido.read()
        nparr_fundo = np.frombuffer(conteudo_fundo, np.uint8)
        img_fundo = cv2.imdecode(nparr_fundo, cv2.IMREAD_COLOR)
        img_fundo = cv2.cvtColor(img_fundo, cv2.COLOR_BGR2RGB)
        imagem_final = tile_texture(img_fundo, h, w)
    else:
        imagem_final = np.full((h, w, 3), (230, 220, 210), dtype=np.uint8)

    # PROCESSA A TEXTURA DE FIO ENVIADA (OU GERA UMA PROVISÓRIA)
    if textura_fio and textura_fio.filename:
        conteudo_fio = await textura_fio.read()
        nparr_fio = np.frombuffer(conteudo_fio, np.uint8)
        img_fio = cv2.imdecode(nparr_fio, cv2.IMREAD_GRAYSCALE)
        textura_fio_gray = tile_texture(img_fio, h, w)
    else:
        y, x = np.mgrid[0:h, 0:w]
        textura_fio_gray = ((np.sin((x + y) * 0.5) + 1) * 127).astype(np.uint8)

    for hex_color in cores_hex:
        rgb_target = np.array(hex_to_rgb(hex_color))
        lower_bound = np.clip(rgb_target - 30, 0, 255)
        upper_bound = np.clip(rgb_target + 30, 0, 255)
        
        mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        
        borda_processada = apply_stitch_effect(mask, rgb_target, textura_fio_gray)
        
        # SOMBRA PROJETADA
        sombra_mask = cv2.GaussianBlur(mask, (15, 15), 0)
        M = np.float32([[1, 0, 5], [0, 1, 8]])
        sombra_mask = cv2.warpAffine(sombra_mask, M, (w, h))
        fator_sombra = 1.0 - (sombra_mask / 255.0) * 0.7 
        for i in range(3):
            imagem_final[:,:,i] = (imagem_final[:,:,i] * fator_sombra).astype(np.uint8)

        idx = mask == 255
        imagem_final[idx] = borda_processada[idx]

    imagem_final_bgr = cv2.cvtColor(imagem_final, cv2.COLOR_RGB2BGR)
    _, encoded_img = cv2.imencode('.png', imagem_final_bgr)
    return Response(content=encoded_img.tobytes(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
