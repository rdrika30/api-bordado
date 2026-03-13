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
        # Mudamos para um modelo mais robusto e detalhista
        output = replicate.run(
            "lucataco/controlnet-canny-xl:67e163b2f29396e94924a1b023158c35b43615170362f6b8a8b16e45f9495140",
            input={
                "image": file.file,
                # PROMPT REFORMULADO PARA FOTORREALISMO EXTREMO
                "prompt": f"RAW photo, extreme close-up macro of a real professional {cor_nome} embroidery patch, individual shiny polyester threads visible, high 3D relief, satin stitch texture, studio lighting with soft shadows, on a textured beige cotton canvas background, 8k uhd, high quality embroidery machine result, photorealistic, cinematic lighting",
                "negative_prompt": "flat, 2d, cartoon, illustration, drawing, painting, blurry, low resolution, plastic texture, messy threads, out of focus, distorted shape",
                "controlnet_conditioning_scale": 1.4, # Aumentado para manter sua logo perfeita
                "guidance_scale": 12.0, # Aumentado para a IA seguir o prompt com mais força
                "num_inference_steps": 40, # Mais passos = mais detalhes
                "image_resolution": 768
            }
        )

        img_url = output[0]
        img_res = requests.get(img_url)
        return Response(content=img_res.content, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
