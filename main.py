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
    """
    Simula um efeito de bordado realista com textura de fio e volume 3D.
    """
    h, w = mask.shape
    
    # 1. Cria a base com a cor escolhida
    base_color = np.zeros((h, w, 3), dtype=np.uint8)
    base_color[mask == 255] = color_rgb

    # 2. Gera a textura dos fios (Ruído Direcional)
    # Criamos um ruído aleatório e aplicamos um borrão para simular fios esticados
    noise = np.random.randint(100, 200, (h, w), dtype=np.uint8)
    k_size = 9 # Espessura do fio
    kernel = np.eye(k_size) / k_size # Matriz diagonal para fios em ângulo
    texture_gray = cv2.filter2D(noise, -1, kernel)
    texture_rgb = cv2.cvtColor(texture_gray, cv2.COLOR_GRAY2RGB)
    
    # Mistura a textura com a cor original (60% cor, 40% textura)
    stitched = cv2.addWeighted(base_color, 0.7, texture_rgb, 0.3, 0)

    # 3. Cria o Volume 3D (Transformada de Distância)
    # Deixa o centro da forma brilhante e as bordas escuras (onde a agulha fura o tecido)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Normaliza a distância para usar como um multiplicador de luz (de 0.3 a 1.2)
    # 0.3 = borda escura, 1.2 = centro brilhante
    cv2.normalize(dist, dist, 0.3, 1.2, cv2.NORM_MINMAX)
    
    # Aplica o brilho de volume na imagem texturizada
    result_float = np.zeros_like(stitched, dtype=np.float32)
    for i in range(3):
        result_float[:,:,i] = stitched[:,:,i] * dist
        
    result_3d = np.clip(result_float, 0, 255).astype(np.uint8)

    # 4. Refina as bordas com micro-sombras (Sobel)
    gray = cv2.cvtColor(result_3d, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    emboss = (sobel_x + sobel_y) * 0.3 # Intensidade da sombra
    emboss = np.clip(emboss, -40, 40).astype(np.int8)

    # Aplica a sombra final
    final_result = result_3d.astype(np.int16)
    for i in range(3):
        final_result[:,:,i] += emboss
        
    final_result = np.clip(final_result, 0, 255).astype(np.uint8)
    
    # Limpa tudo que estiver fora da máscara
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
