import face_recognition
import json
import numpy as np

# load encodings from disk
with open("encodings.json") as f:
    stored = json.load(f)

known_names = list(stored.keys())
known_encodings = [np.array(stored[name]) for name in known_names]

# pick an image to test
# unknown_image_path = r"faces\opama.png"
unknown_image_path = r"uploaded_images\2025-09-30 12-43-51.316562"
unknown_image = face_recognition.load_image_file(unknown_image_path)
unknown_encodings = face_recognition.face_encodings(unknown_image)

if not unknown_encodings:
    print("No face found in test image.")
else:
    unknown_encoding = unknown_encodings[0]
    # compare
    results = face_recognition.compare_faces(known_encodings, unknown_encoding)
    print(results)

    # print matches
    for name, matched in zip(known_names, results):
        if matched:
            print(f"Match found: {name}")
