import requests

url = "http://127.0.0.1:8000/recognize"
files = {"file": open(r"faces\alexandre.jpg", "rb")}
resp = requests.post(url, files=files)
print(resp.json())
