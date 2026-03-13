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
        # Usando um modelo ESPECIALIZADO apenas em bordado real (SDXL Embroidery)
        output = replicate.run(
            "fofr/sdxl-embroidery:16960f2746a51ed8148b046e7f10b75960098df90393699478f6c59b2d8f7602",
            input={
                "image": file.file,
                "prompt": f"embroidery of {cor_nome} logo, detailed stitching, realistic threads, high relief, macro photography, 8k, on fabric background",
                "negative_prompt": "low quality, blurry, flat, plastic, drawing",
                "strength": 0.8, # Define o quanto a IA pode mudar a imagem para parecer fio
                "guidance_scale": 7.5,
                "num_inference_steps": 40
            }
        )

        img_url = output[0]
        img_res = requests.get(img_url)
        return Response(content=img_res.content, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
