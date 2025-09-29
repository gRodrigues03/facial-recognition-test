import face_recognition
import json

# 1. pick a file and a name for the person
image_path = r"faces\opama.webp"
# image_path = r"faces\blue-pen.png"
person_name = "opama"

# 2. load or create the storage dict
try:
    with open("encodings.json") as f:
        stored = json.load(f)
except FileNotFoundError:
    stored = {}

# 3. compute encoding
image = face_recognition.load_image_file(image_path)
encodings = face_recognition.face_encodings(image)
if not encodings:
    raise RuntimeError("No face found in the image!")
encoding = encodings[0]

# 4. save it
stored[person_name] = encoding.tolist()
with open("encodings.json", "w") as f:
    json.dump(stored, f)

print("Saved encoding for", person_name)
