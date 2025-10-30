# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 21:49:56 2025

@author: anton
"""


from fastapi import FastAPI, UploadFile, Form
from typing import List
from transformers import CLIPProcessor ,CLIPModel, BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from PIL import Image
import torch
import io

# Para features preentrenados
from torchvision import models, transforms
import numpy as np


# Inicializar FastAPI
app = FastAPI()

# Usar CPU (si tienes GPU, cambia a "cuda")
device = torch.device("cpu")

# ============================
# Cargar modelos
# ============================

# CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Traducción inglés -> español
translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es").to(device)

# BLIP para captions
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# ResNet50 como extractor de características
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # quitar capa de clasificación
resnet.eval()

# Transformaciones para ResNet
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================
# Funciones auxiliares
# ============================

def extract_features_resnet(image: Image.Image) -> np.ndarray:
    """Extrae features de una imagen usando ResNet50 preentrenada"""
    img_t = resnet_transform(image).unsqueeze(0)  # batch=1
    with torch.no_grad():
        feats = resnet(img_t).squeeze().numpy()
    return feats

def compute_photo_metrics(features: np.ndarray) -> dict:
    """Convierte features en métricas interpretables"""
    clarity = float(abs(features.mean()))        # media = "claridad"
    harmony = float(abs(features.std()))         # desviación = "armonía visual"
    balance = float(abs(features.max() - features.min()))  # rango = "balance"
    
    # Escalar a rango 0-10
    def scale(x):
        return round(10 * (x / (x + 1e-6)), 2) if x > 0 else 0.0
    
    return {
        "clarity": scale(clarity),
        "harmony": scale(harmony),
        "balance": scale(balance)
    }

def clip_similarity(image_bytes: bytes, prompt: str) -> float:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    return probs[0][0].item()

def translate_en_to_es(text):
    inputs = translator_tokenizer(text, return_tensors="pt").to(device)
    translated = translator_model.generate(**inputs)
    output = translator_tokenizer.decode(translated[0], skip_special_tokens=True)

  # Corrección manual de errores frecuentes
    output = output.replace("arafe", "aparece")
    output = output.replace("Arafed", "aparece")
    output = output.replace("Arafe", "aparece")
    output = output.replace("Araffe", "aparece")
    output = output.replace("araffe", "aparece")

  # Asegurar que la primera letra es mayúscula
    if output:
      output = output[0].upper() + output[1:]
      
    return output

def generate_feedback(image: Image.Image, prompt: str) -> str:
    """Devuelve feedback enriquecido combinando BLIP y CLIP"""

    candidate_feedbacks = [
        "La foto tiene una iluminación equilibrada que resalta los detalles.",
        "El encuadre centra la atención en el sujeto principal de forma clara.",
        "Los colores son vivos y transmiten una sensación agradable.",
        "La composición transmite profesionalidad y cuidado estético.",
        "La imagen coincide fielmente con el concepto solicitado.",
        "El fondo es limpio y no distrae la atención del sujeto.",
        "La perspectiva utilizada da profundidad y atractivo visual.",
        "El estilo de la foto resulta elegante y bien estructurado.",
        "La nitidez y claridad de la imagen destacan frente a las demás.",
        "La postura del sujeto transmite seguridad y naturalidad.",
        "El contraste de luces y sombras aporta dinamismo.",
        "El ambiente general de la foto es armonioso y coherente.",
        "El rostro y expresiones están bien capturados, transmitiendo emociones.",
        "La proporción entre sujeto y fondo está bien balanceada.",
        "El uso del color refuerza la sensación de profesionalidad.",
        "El retrato está bien centrado y enfocado, generando impacto.",
        "Los detalles del entorno acompañan sin restar protagonismo.",
        "La foto transmite confianza y cercanía al observador.",
        "La calidad técnica es alta, con buena definición en las texturas.",
        "La iluminación natural favorece el aspecto del sujeto.",
        "El estilo de la imagen se adapta al propósito profesional.",
        "La simplicidad del fondo ayuda a reforzar el mensaje principal.",
        "El ángulo elegido aporta originalidad sin perder claridad.",
        "La fotografía genera una sensación de armonía visual.",
        "El sujeto se integra de manera natural en la escena.",
        "La composición aprovecha el espacio de manera equilibrada.",
        "El enfoque resalta lo más importante de la imagen.",
        "La combinación de colores transmite seriedad y profesionalismo.",
        "La imagen refleja autenticidad y espontaneidad.",
        "El conjunto de elementos crea un resultado atractivo y convincente."
    ]

    # Caption con BLIP
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    generated_description = processor.decode(out[0], skip_special_tokens=True)

    # Evaluar feedback con CLIP
    inputs_fb = clip_processor(text=candidate_feedbacks, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs_fb = clip_model(**inputs_fb)
        probs_fb = outputs_fb.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    best_feedback = candidate_feedbacks[probs_fb.argmax()]

    generated_description_es = translate_en_to_es(generated_description)
    return f"{generated_description_es}. {best_feedback}"


# ============================
# Endpoint principal
# ============================

@app.post("/rank-photos/")
async def rank_photos(
    files: List[UploadFile],
    prompt: str = Form("Foto con mejor armonía visual")
):
    images = []
    filenames = []
    metrics_list = []

    # Cargar imágenes y extraer métricas
    for file in files:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(image)
        filenames.append(file.filename)

        # Extraer features con ResNet y calcular métricas
        feats = extract_features_resnet(image)
        metrics = compute_photo_metrics(feats)
        metrics_list.append(metrics)

    # Procesar similitud con CLIP
    inputs = clip_processor(text=[prompt], images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image

    # Crear resultados con similitud + métricas
    results = []
    for i, fname in enumerate(filenames):
        score = logits_per_image[i][0].item()
        results.append({
            "filename": fname,
            "similarity": score,
            "metrics": metrics_list[i]
        })

    # Normalizar similitudes a rango 0-10
    min_score = min(r["similarity"] for r in results)
    max_score = max(r["similarity"] for r in results)
    eps = 1e-6
    for r in results:
        normalized_score = (r["similarity"] - min_score) / (max_score - min_score + eps)
        r["similarity"] = round(normalized_score * 10, 2)

    # Ordenar
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    # Feedback de la mejor imagen
    best_image = images[filenames.index(results[0]["filename"])]
    feedback = generate_feedback(best_image, prompt)

    return {"prompt": prompt, "ranking": results, "feedback": feedback}











"""
from fastapi import FastAPI, UploadFile, Form
from typing import List
from transformers import CLIPProcessor ,CLIPModel, BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from PIL import Image
import torch
import io
import numpy as np
import cv2
import mediapipe as mp

# Inicializar FastAPI
app = FastAPI()

# Usar CPU (si tienes GPU, cambia a "cuda")
device = torch.device("cpu")

# Cargar CLIP localmente (esto se hace solo una vez al inicio)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Cargar modelo de traducción inglés -> español
translator_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
translator_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es").to(device)

# Cargar el procesador y el modelo BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# Mediapipe
mp_face_mesh = mp.solutions.face_mesh


def clip_similarity(image_bytes: bytes, prompt: str) -> float:
    "Calcula similitud imagen-texto usando CLIP local""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # similitud imagen-texto
        probs = logits_per_image.softmax(dim=1)

    return probs[0][0].item()  # devuelve un float entre 0 y 1


def translate_en_to_es(text):
    inputs = translator_tokenizer(text, return_tensors="pt").to(device)
    translated = translator_model.generate(**inputs)
    return translator_tokenizer.decode(translated[0], skip_special_tokens=True)


def generate_feedback(image: Image.Image, prompt: str) -> str:
    "Devuelve un feedback enriquecido para la imagen, combinando una descripción generada y un comentario predefinido.""

    # Feedbacks predeterminados
    candidate_feedbacks = [
        "La foto tiene una iluminación equilibrada que resalta los detalles.",
        "El encuadre centra la atención en el sujeto principal de forma clara.",
        "Los colores son vivos y transmiten una sensación agradable.",
        "La composición transmite profesionalidad y cuidado estético.",
        "La imagen coincide fielmente con el concepto solicitado.",
        "El fondo es limpio y no distrae la atención del sujeto.",
        "La perspectiva utilizada da profundidad y atractivo visual.",
        "El estilo de la foto resulta elegante y bien estructurado.",
        "La nitidez y claridad de la imagen destacan frente a las demás.",
        "La postura del sujeto transmite seguridad y naturalidad.",
        "El contraste de luces y sombras aporta dinamismo.",
        "El ambiente general de la foto es armonioso y coherente.",
        "El rostro y expresiones están bien capturados, transmitiendo emociones.",
        "La proporción entre sujeto y fondo está bien balanceada.",
        "El uso del color refuerza la sensación de profesionalidad.",
        "El retrato está bien centrado y enfocado, generando impacto.",
        "Los detalles del entorno acompañan sin restar protagonismo.",
        "La foto transmite confianza y cercanía al observador.",
        "La calidad técnica es alta, con buena definición en las texturas.",
        "La iluminación natural favorece el aspecto del sujeto.",
        "El estilo de la imagen se adapta al propósito profesional.",
        "La simplicidad del fondo ayuda a reforzar el mensaje principal.",
        "El ángulo elegido aporta originalidad sin perder claridad.",
        "La fotografía genera una sensación de armonía visual.",
        "El sujeto se integra de manera natural en la escena.",
        "La composición aprovecha el espacio de manera equilibrada.",
        "El enfoque resalta lo más importante de la imagen.",
        "La combinación de colores transmite seriedad y profesionalismo.",
        "La imagen refleja autenticidad y espontaneidad.",
        "El conjunto de elementos crea un resultado atractivo y convincente."
    ]

    # Generar una descripción de la imagen con BLIP
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    generated_description = processor.decode(out[0], skip_special_tokens=True)

    # Procesar las frases de feedback contra la imagen
    inputs_fb = clip_processor(
        text=candidate_feedbacks,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs_fb = clip_model(**inputs_fb)
        logits_per_image = outputs_fb.logits_per_image
        probs_fb = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    best_feedback = candidate_feedbacks[probs_fb.argmax()]

    # Combinar la descripción generada con el feedback seleccionado
    generated_description_es = translate_en_to_es(generated_description)
    return f"{generated_description_es}. {best_feedback}"


def analyze_face_technical(image_bytes: bytes):
    "Analiza rasgos técnicos del rostro con mediapipe y devuelve un score.""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return {"symmetry": 0.0, "proportion": 0.0, "ratio": 0.0, "score": 0.0}

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = img_np.shape
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

        # Simetría facial aproximada
        mid_x = w / 2
        left = points[points[:,0] < mid_x]
        right = points[points[:,0] > mid_x]
        symmetry = 1.0 / (1.0 + np.mean(np.abs(left[:,0].mean() - (w - right[:,0].mean()))))

        # Proporción vertical: ojos-nariz-boca-barbilla
        eye_y = np.mean(points[159:161,1])
        nose_y = points[1,1]
        mouth_y = np.mean(points[13:14,1])
        chin_y = points[152,1]
        top = nose_y - eye_y
        mid = mouth_y - nose_y
        low = chin_y - mouth_y
        proportion = 1.0 / (1.0 + abs((top-mid)+(mid-low)))

        # Ratio ancho/alto
        face_width = np.max(points[:,0]) - np.min(points[:,0])
        face_height = np.max(points[:,1]) - np.min(points[:,1])
        ratio = face_width / (face_height+1e-6)

        # Score final normalizado 0–10
        score = float((symmetry + proportion + (1.0/(1+abs(ratio-0.75)))) / 3 * 10)

        return {
            "symmetry": float(symmetry*10),
            "proportion": float(proportion*10),
            "ratio": float(ratio),
            "score": round(score, 2)
        }


@app.post("/rank-photos/")
async def rank_photos(
    files: List[UploadFile],
    prompt: str = Form("Foto profesional")
):
    images = []
    filenames = []
    raw_bytes = []

    # Cargar todas las imágenes en memoria
    for file in files:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        images.append(image)
        filenames.append(file.filename)
        raw_bytes.append(img_bytes)

    # Procesar todas las imágenes juntas contra el prompt
    inputs = clip_processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # [num_images, num_texts]

    # Crear resultados
    results = []
    for i, fname in enumerate(filenames):
        score_clip = logits_per_image[i][0].item()

        # Normalizar CLIP entre 0–10
        score_clip_norm = (score_clip - logits_per_image.min().item()) / (
            logits_per_image.max().item() - logits_per_image.min().item() + 1e-6
        ) * 10

        # Análisis facial técnico
        face_analysis = analyze_face_technical(raw_bytes[i])
        face_score = face_analysis["score"]

        # Media de ambos scores
        final_score = (score_clip_norm + face_score) / 2

        results.append({
            "filename": fname,
            "similarity": round(score_clip_norm, 2),
            "face_score": face_score,
            "final_score": round(final_score, 2),
            "face_analysis": face_analysis
        })

    # Ordenar por score final
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    best_image = images[filenames.index(results[0]["filename"])]
    feedback = generate_feedback(best_image, prompt)
    
    return {"prompt": prompt, "ranking": results, "feedback": feedback}"""

