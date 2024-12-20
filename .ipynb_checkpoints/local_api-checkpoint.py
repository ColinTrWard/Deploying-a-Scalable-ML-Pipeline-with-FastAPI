import json
import requests

# Send a GET request to the API
r = requests.get("http://127.0.0.1:8000/")

# Print the status code
print(f"GET request status code: {r.status_code}")

# Print the welcome message
if r.status_code == 200:
    print(f"GET response: {r.json()}")

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request to the API
r = requests.post("http://127.0.0.1:8000/data/", json=data)

# Print the status code
print(f"POST request status code: {r.status_code}")

# Print the result
if r.status_code == 200:
    print(f"POST response: {r.json()}")
