import requests

url = "http://127.0.0.1:8000/docs#/default/predict_predict_post"
files = {'file': open("dog.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
