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
    
    # Prompt técnico que aciona o "conhecimento" do modelo especializado
    prompt_especialista = (
        f"A professional embroidery patch of a logo with colors {', '.join(cores_hex)}. "
        "FLUX_EMBROIDERY style, thick satin stitches, directional embroidery paths, "
        "3D high relief, individual thread texture visible, shiny polyester thread, "
        "slight imperfections at the edges, studio lighting, macro photography, 8k, "
        "on a heavy textured canvas background."
    )

    try:
        # Este modelo é uma 'fina sintonia' focada 100% em realismo de costura
        output = replicate.run(
            "lucataco/flux-dev-lora:a88096a6", # Base Flux com suporte a LoRA
            input={
                "image": file.file,
                "prompt": prompt_especialista,
                "lora_url": "https://replicate.delivery/pbxt/JzF.../embroidery_style.safetensors", # Modelo de bordado real
                "control_image": file.file,
                "control_type": "canny",
                "num_inference_steps": 50,
                "guidance_scale": 4.5,
                "prompt_strength": 0.9,
                "controlnet_conditioning_scale": 0.85
            }
        )

        img_url = output[0]
        img_res = requests.get(img_url)
        return Response(content=img_res.content, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
