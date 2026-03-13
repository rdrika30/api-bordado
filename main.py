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
    
    # O SEGREDO ESTÁ NESTE PROMPT ULTRA-DETALHADO
    prompt_final = (
        f"Macro photography of a physical custom embroidery patch. "
        f"The logo colors ({', '.join(cores_hex)}) are made of thick, shiny trilobal polyester threads. "
        "High-density satin stitching with visible directional thread paths. "
        "Extreme 3D embossed relief, threads overlapping at the edges, micro-shadows between each stitch. "
        "Slightly irregular borders showing individual thread fibers piercing a heavy blue denim fabric. "
        "Professional studio lighting, 8k resolution, cinematic macro, hyper-realistic texture."
    )

    try:
        # Usando o FLUX.1 [DEV] - O modelo mais potente para texturas do mundo
        output = replicate.run(
            "black-forest-labs/flux-dev",
            input={
                "image": file.file,
                "prompt": prompt_final,
                "control_image": file.file, # Força a IA a seguir a sua logo
                "control_type": "canny",
                "negative_prompt": "flat, 2d, vector, smooth, plastic, cartoon, drawing, clean edges, low quality",
                "guidance_scale": 25.0, # Muita força no prompt
                "num_inference_steps": 50,
                "extra_control": "canny",
                "controlnet_conditioning_scale": 0.95 
            }
        )

        img_url = output[0]
        img_res = requests.get(img_url)
        return Response(content=img_res.content, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
