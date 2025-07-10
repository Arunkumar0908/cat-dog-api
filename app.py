from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()
model = load_model("cat_dog_model.h5")

IMG_SIZE = 128

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)
    return {"prediction": label, "confidence": round(confidence, 3)}
