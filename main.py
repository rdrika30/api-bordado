import os
import replicate
import ast
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/aplicar_bordado/")
async def aplicar_bordado(file: UploadFile = File(...), cores_selecionadas: str = Form(...)):
    # 1. Converte a string de cores de volta para uma lista
    cores_hex = ast.literal_eval(cores_selecionadas)
    
    # Criamos um prompt que descreve todas as cores para a IA
    descricao_cores = ", ".join(cores_hex)
    
    contents = await file.read()
    
    try:
        # Usamos o FLUX-FILL que é especialista em respeitar as cores originais 
        # e adicionar textura realista por cima.
        output = replicate.run(
            "black-forest-labs/flux-fill",
            input={
                "image": file.file,
                "prompt": f"Professional photorealistic embroidery patch. The colors {descricao_cores} are made of thick silk threads. Satin stitch texture, 3D relief, macro photography, visible thread fibers, studio lighting, on a canvas background.",
                "negative_prompt": "plastic, flat, 2d, cartoon, drawing, blurry",
                "guidance_scale": 25,
                "num_inference_steps": 50,
                "prompt_strength": 0.9,
                "image_resolution": 768
            }
        )

        img_url = output[0]
        img_res = requests.get(img_url)
        return Response(content=img_res.content, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
