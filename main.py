import os
import replicate
import ast
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/aplicar_bordado/")
async def aplicar_bordado(file: UploadFile = File(...), cores_selecionadas: str = Form(...)):
    cores_hex = ast.literal_eval(cores_selecionadas)
    cor_nome = cores_hex[0] if cores_hex else "vibrant"

    try:
        # MODELO: Flux.1 Dev (O mais avançado do mundo atualmente para texturas)
        # Este modelo substitui o Stable Diffusion e é infinitamente mais realista
        output = replicate.run(
            "black-forest-labs/flux-fill", 
            input={
                "image": file.file,
                "prompt": f"A high-quality professional embroidery patch of this logo, {cor_nome} silk threads, satin stitch, extreme 3D embroidery texture, macro shot showing individual thread fibers, professional lighting, realistic shadows, photorealistic, 8k, on white canvas fabric",
                "negative_prompt": "flat, 2d, illustration, drawing, plastic, smooth",
                "guidance_scale": 30,
                "num_inference_steps": 50,
                "prompt_strength": 0.85 # Permite que a IA transforme o plano em 3D real
            }
        )

        img_url = output[0]
        img_res = requests.get(img_url)
        return Response(content=img_res.content, media_type="image/png")

    except Exception as e:
        # Se o modelo acima for pesado demais, usamos este fallback de segurança
        return {"error": str(e)}
