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
    # 1. Pega a cor que você escolheu no site
    cores_hex = ast.literal_eval(cores_selecionadas)
    cor_nome = cores_hex[0] if cores_hex else "vibrant"

    # 2. Envia para a Replicate (IA)
    # Este modelo usa ControlNet para manter o formato exato da sua logo
    try:
        output = replicate.run(
            "lucataco/controlnet-canny-xl:67e163b2f29396e94924a1b023158c35b43615170362f6b8a8b16e45f9495140",
            input={
                "image": file.file,
                "prompt": f"Extreme macro photography of a professional {cor_nome} embroidery patch, thick satin stitch threads, 3D high relief, realistic sewing texture, studio lighting, on beige canvas fabric background, 8k resolution",
                "negative_prompt": "cartoon, drawing, flat, plastic, blurry, low quality, thin lines",
                "controlnet_conditioning_scale": 1.2,
                "guidance_scale": 7.5,
                "num_inference_steps": 30
            }
        )

        # 3. Baixa a imagem gerada pela IA e entrega para o seu index.html
        img_url = output[0]
        img_res = requests.get(img_url)
        return Response(content=img_res.content, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
