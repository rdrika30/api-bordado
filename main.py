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
    cores_hex = ast.literal_eval(cores_selecionadas)
    # Criamos um prompt ultra-descritivo focado em TEXTURA e IMPERFEIÇÃO
    prompt_text = (
        f"Extreme macro photo of a real physical embroidery patch, colors: {', '.join(cores_hex)}. "
        "Thick satin stitch, visible interlocking polyester threads, high 3D embroidery relief, "
        "rough thread edges, shiny silk texture, studio lighting with deep shadows between threads, "
        "photorealistic, 8k, highly detailed textile machinery result, on heavy denim fabric background"
    )

    try:
        # Usando o modelo de DEPTH (Profundidade) para garantir o relevo 3D
        output = replicate.run(
            "stability-ai/sdxl-controlnet-depth-paints-generator:af15086d", 
            input={
                "image": file.file,
                "prompt": prompt_text,
                "negative_prompt": "flat, 2d, logo, vector, smooth, plastic, drawing, painting, blurry, thin lines, clean edges",
                "num_inference_steps": 50,
                "guidance_scale": 15,
                "controlnet_conditioning_scale": 0.8, # Deixa a IA "vazar" um pouco a linha para parecer fio real
                "strength": 0.9 # Força a IA a ignorar a textura lisa original
            }
        )

        img_url = output[0]
        img_res = requests.get(img_url)
        return Response(content=img_res.content, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
