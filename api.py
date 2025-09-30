import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import face_recognition
import json
import numpy as np
import shutil
import os
import cv2

app = FastAPI()

# Load known encodings
ENCODINGS_FILE = "encodings.json"
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE) as f:
        stored = json.load(f)
else:
    stored = {}

known_names = list(stored.keys())
known_encodings = [np.array(stored[name]) for name in known_names]

# Ensure directory exists to save uploaded images
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def preprocess_image(file_path: str, max_dim: int = 800):
    """Light preprocessing: resize large images and convert to RGB."""
    image = cv2.imread(file_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        timestamp = str(datetime.datetime.now()).replace(':', '-')
        saved_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        unknown_image = preprocess_image(file_path)
        if unknown_image is None:
            return JSONResponse(
                content={"match": False, "message": "Failed to read image"},
                status_code=400
            )

        unknown_encodings = face_recognition.face_encodings(unknown_image)
        if not unknown_encodings:
            face_locations = face_recognition.face_locations(unknown_image, model="cnn")
            if not face_locations:
                return {"match": False, "message": "Nenhum rosto encontrado na imagem"}
            unknown_encodings = face_recognition.face_encodings(unknown_image, face_locations)

        unknown_encoding = unknown_encodings[0]
        results = face_recognition.compare_faces(known_encodings, unknown_encoding)
        matches = [name for name, matched in zip(known_names, results) if matched]

        return {"match": bool(matches), "matches": matches, "message": f"Rostos encontrados: {', '.join(matches)}"}

    except Exception as e:
        return {"match": False, "message": f"Error processing image: {str(e)}"}

@app.post("/insert")
async def insert_face(name: str, file: UploadFile = File(...)):
    try:
        # Save uploaded image
        timestamp = str(datetime.datetime.now()).replace(':', '-')
        saved_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        image = preprocess_image(file_path)
        if image is None:
            return JSONResponse(content={"success": False, "message": "Failed to read image"}, status_code=400)

        encodings = face_recognition.face_encodings(image)
        if not encodings:
            return {"success": False, "message": "No face detected in image"}

        # Store encoding
        stored[name] = encodings[0].tolist()
        with open(ENCODINGS_FILE, "w") as f:
            json.dump(stored, f)

        # Update memory cache
        known_names.append(name)
        known_encodings.append(encodings[0])

        return {"success": True, "saved_file": saved_filename}

    except Exception as e:
        return {"success": False, "message": str(e)}
