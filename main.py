from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Middleware CORS (boleh kamu ganti dengan domain asli saat produksi)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti dengan domain frontend saat production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model saat server start
model = tf.keras.models.load_model("model.keras")

# Sesuaikan ini dengan label yang digunakan di modelmu
class_names = ["sehat", "agak_layu", "sangat_layu", "lainnya"]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # jadi (1, 224, 224, 3)
    return image_array

@app.get("/")
def root():
    return {"message": "API siap digunakan 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    predictions = model.predict(input_tensor)
    predicted_index = int(np.argmax(predictions))
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(predictions))
    return {
        "class": predicted_label,
        "confidence": confidence
    }
