import requests

url = "http://127.0.0.1:8000/recognize"
files = {"file": open(r"uploaded_images\2025-09-30 12-50-53.088346.jpg", "rb")}
resp = requests.post(url, files=files)
print(resp.json())
