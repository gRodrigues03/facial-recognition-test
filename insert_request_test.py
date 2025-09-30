import requests

url = "http://127.0.0.1:8000/insert"
files = {"file": open(r"faces\Silveira.jpg", "rb")}
params = {"name": "Silveira"}  # query parameter

resp = requests.post(url, files=files, params=params)
print(resp.json())