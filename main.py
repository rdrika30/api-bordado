import os
import replicate
import ast
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

# Configure sua chave da Replicate nas variáveis de ambiente do Render
# Ou cole aqui temporariamente: os.environ["REPLICATE_API_TOKEN"] = "sua_chave_aqui"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/aplicar_bordado/")
async def aplicar_bordado(file: UploadFile = File(...), cores_selecionadas: str = Form(...)):
    # 1. Lemos a imagem e a cor
    cores_hex = ast.literal_eval(cores_selecionadas)
    cor_alvo = cores_hex[0] # Vamos focar na primeira cor selecionada para o teste
    
    contents = await file.read()
    # Salva temporariamente para enviar para a IA
    with open("input.png", "wb") as f:
        f.write(contents)

    # 2. Chamada para a IA (Stable Diffusion + ControlNet)
    # Este modelo abaixo é especializado em seguir a forma da imagem (Canny Edge)
    output = replicate.run(
        "jagadeesh-geetha/controlnet-canny:7f16bb31", # Exemplo de modelo ControlNet
        input={
            "image": open("input.png", "rb"),
            "prompt": f"Professional realistic {cor_alvo} embroidery patch, satin stitch, high relief, silk thread texture, 8k macro photography, canvas background",
            "num_samples": 1,
            "image_resolution": "512",
            "low_threshold": 100,
            "high_threshold": 200,
            "ddim_steps": 20,
            "scale": 9
        }
    )

    # 3. A IA devolve uma URL da imagem pronta
    # O Python baixa essa imagem e entrega para o seu site
    import requests
    img_data = requests.get(output[0]).content
    return Response(content=img_data, media_type="image/png")

# O restante das funções (paleta, etc) continuam iguais...
