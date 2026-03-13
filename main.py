from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from sklearn.cluster import KMeans
import uvicorn
import ast

app = FastAPI(title="API Bordado Pro")

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

def gerar_fundo_tecido(h, w, cor_base=(230, 220, 210)):
    """Gera uma textura que imita as tramas de um tecido canvas/sarja."""
    fundo = np.full((h, w, 3), cor_base, dtype=np.uint8)
    
    # Cria ruído aleatório
    ruido = np.random.randint(0, 25, (h, w), dtype=np.uint8)
    
    # Estica o ruído na horizontal e vertical para criar a "trama"
    kernel_h = np.ones((1, 5)) / 5
    kernel_v = np.ones((5, 1)) / 5
    trama_h = cv2.filter2D(ruido, -1, kernel_h)
    trama_v = cv2.filter2D(ruido, -1, kernel_v)
    
    # Mistura as tramas
    trama_tecido = cv2.addWeighted(trama_h, 0.5, trama_v, 0.5, 0)
    trama_tecido = cv2.cvtColor(trama_tecido, cv2.COLOR_GRAY2RGB)
    
    # Escurece levemente o fundo com a trama
    return cv2.subtract(fundo, trama_tecido)

def apply_stitch_effect(mask, color_rgb):
    """
    Simula preenchimento de bordado "Soft-Relief" com fios senoidais suaves
    e volume 3D global ('travesseiro').
    """
    h, w = mask.shape
    
    # === PASSO 1: Fios Senoidais Suaves ===
    # Geramos uma trama senoidal para evitar linhas retas e duras.
    y, x = np.mgrid[0:h, 0:w]
    # 'freq_fios' dita a espessura. Menor = mais grosso/suave.
    freq_fios = 0.8
    perfil_fios = (np.sin((x + y) * freq_fios) + 1) / 2.0 
    
    stitched = np.zeros((h, w, 3), dtype=np.uint8)
    cor_base = np.array(color_rgb, dtype=np.int16)
    
    # Calcula as cores para fios gordinhos e suaves (Soft Gloss)
    for i in range(3):
        # fresta profunda (0.35) a topo gordinho (1.2)
        intensidade_luz = 0.35 + (perfil_fios * 0.85) 
        stitched[:,:,i] = np.clip(cor_base[i] * intensidade_luz, 0, 255)
    
    # === PASSO 2: Volume 3D Global (Pillow Effect) ===
    # Calcula a distância do centro da forma.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Normaliza a distância para usar como um multiplicador de luz (relevo).
    # O centro fica mais alto e brilhante, as bordas mais baixas e escuras.
    cv2.normalize(dist, dist, 0.45, 1.35, cv2.NORM_MINMAX) # Ampla faixa para relevo
    
    # Aplica o relevo 3D global na imagem texturizada
    result_float = np.zeros_like(stitched, dtype=np.float32)
    for i in range(3):
        result_float[:,:,i] = stitched[:,:,i] * dist
        
    result_3d = np.clip(result_float, 0, 255).astype(np.uint8)

    # === PASSO 3: Micro-sombras e Crocância ===
    # Melhora o contraste local para que os fios 'saltem' individualmente.
    gray = cv2.cvtColor(result_3d, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    emboss = (sobel_x + sobel_y) * 0.6 # Intensidade da crocância
    emboss = np.clip(emboss, -60, 60).astype(np.int8)

    final_result = result_3d.astype(np.int16)
    for i in range(3):
        final_result[:,:,i] += emboss
        
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)
    
    # Limpa o que vazou da máscara
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
    
    # 1. Cria o fundo de tecido realista
    imagem_final = gerar_fundo_tecido(h, w)
    
    for hex_color in cores_hex:
        rgb_target = np.array(hex_to_rgb(hex_color))
        lower_bound = np.clip(rgb_target - 30, 0, 255)
        upper_bound = np.clip(rgb_target + 30, 0, 255)
        
        mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        
        # 2. Gera o bordado com volume
        borda_processada = apply_stitch_effect(mask, rgb_target)
        
        # 3. GERA A SOMBRA PROJETADA (Drop Shadow)
        # Borra a máscara para criar uma sombra suave
        sombra_mask = cv2.GaussianBlur(mask, (15, 15), 0)
        # Move a sombra 5 pixels para a direita e 8 para baixo
        M = np.float32([[1, 0, 5], [0, 1, 8]])
        sombra_mask = cv2.warpAffine(sombra_mask, M, (w, h))
        
        # Aplica a sombra escurecendo o fundo do tecido
        fator_sombra = 1.0 - (sombra_mask / 255.0) * 0.6 # Escurece até 60%
        for i in range(3):
            imagem_final[:,:,i] = (imagem_final[:,:,i] * fator_sombra).astype(np.uint8)

        # 4. Cola o bordado volumoso por cima de tudo
        idx = mask == 255
        imagem_final[idx] = borda_processada[idx]

    imagem_final_bgr = cv2.cvtColor(imagem_final, cv2.COLOR_RGB2BGR)
    _, encoded_img = cv2.imencode('.png', imagem_final_bgr)
    return Response(content=encoded_img.tobytes(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





