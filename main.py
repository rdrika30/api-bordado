from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from sklearn.cluster import KMeans
import uvicorn
import ast
import os

app = FastAPI(title="API Bordado Pro - Texturas Reais")

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

def carregar_fundo(h, w):
    """Carrega a foto real do tecido de fundo, ou gera um provisório."""
    caminho = "fundo_tecido.jpg"
    if os.path.exists(caminho):
        fundo = cv2.imread(caminho, cv2.IMREAD_COLOR)
        fundo = cv2.cvtColor(fundo, cv2.COLOR_BGR2RGB)
        return tile_texture(fundo, h, w)
    else:
        # Fundo provisório bege se a imagem não existir
        return np.full((h, w, 3), (230, 220, 210), dtype=np.uint8)

def carregar_textura_fio(h, w):
    """Carrega a foto real do fio (em preto e branco), ou gera provisória."""
    caminho = "textura_fio.jpg"
    if os.path.exists(caminho):
        textura = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        return tile_texture(textura, h, w)
    else:
        # Fio provisório com listras simples se a imagem não existir
        y, x = np.mgrid[0:h, 0:w]
        textura_temp = ((np.sin((x + y) * 0.5) + 1) * 127).astype(np.uint8)
        return textura_temp

def apply_stitch_effect(mask, color_rgb, textura_fio_gray):
    """
    Efeito Estilo Photoshop: Pinta a foto da textura e aplica Chanfro 3D.
    """
    h, w = mask.shape
    
    # 1. Multiplicação de Cor (Pinta a textura cinza com a cor escolhida)
    # Transforma a textura de 0-255 para 0.0-1.0 para agir como um mapa de luz
    tex_float = textura_fio_gray.astype(np.float32) / 255.0
    
    stitched = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        # Multiplica a cor sólida pela luz/sombra da textura fotográfica
        stitched[:,:,i] = np.clip(color_rgb[i] * tex_float, 0, 255)

    # 2. Efeito Chanfro e Entalhe (Bevel & Emboss do Photoshop)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    # Borda mais escura (0.4) descendo para o tecido, centro normal (1.1)
    cv2.normalize(dist, dist, 0.4, 1.1, cv2.NORM_MINMAX)
    
    result_float = np.zeros_like(stitched, dtype=np.float32)
    for i in range(3):
        result_float[:,:,i] = stitched[:,:,i] * dist
        
    result_3d = np.clip(result_float, 0, 255).astype(np.uint8)

    # 3. Micro-sombras para destacar as ranhuras da foto real
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
async def aplicar_bordado(file: UploadFile = File(...), cores_selecionadas: str = Form(...)):
    cores_hex = ast.literal_eval(cores_selecionadas)
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w, _ = img_rgb.shape
    
    # PREPARAÇÃO DAS IMAGENS REAIS
    imagem_final = carregar_fundo(h, w)
    textura_fio_gray = carregar_textura_fio(h, w)
    
    for hex_color in cores_hex:
        rgb_target = np.array(hex_to_rgb(hex_color))
        lower_bound = np.clip(rgb_target - 30, 0, 255)
        upper_bound = np.clip(rgb_target + 30, 0, 255)
        
        mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        
        # GERA O BORDADO USANDO A FOTO
        borda_processada = apply_stitch_effect(mask, rgb_target, textura_fio_gray)
        
        # SOMBRA PROJETADA (Drop Shadow)
        sombra_mask = cv2.GaussianBlur(mask, (15, 15), 0)
        M = np.float32([[1, 0, 5], [0, 1, 8]])
        sombra_mask = cv2.warpAffine(sombra_mask, M, (w, h))
        
        fator_sombra = 1.0 - (sombra_mask / 255.0) * 0.7 # Sombra forte
        for i in range(3):
            imagem_final[:,:,i] = (imagem_final[:,:,i] * fator_sombra).astype(np.uint8)

        # COLA O BORDADO
        idx = mask == 255
        imagem_final[idx] = borda_processada[idx]

    imagem_final_bgr = cv2.cvtColor(imagem_final, cv2.COLOR_RGB2BGR)
    _, encoded_img = cv2.imencode('.png', imagem_final_bgr)
    return Response(content=encoded_img.tobytes(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
