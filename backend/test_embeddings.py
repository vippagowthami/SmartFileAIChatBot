import requests
import json

def test_embeddings():
    url = "http://localhost:11434/api/embed"
    payload = {
        "model": "all-minilm",
        "input": "This is a test document about CORBA."
    }
    try:
        print(f"Connecting to {url}...")
        r = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            print(f"Embedding length: {len(r.json().get('embeddings', [[]])[0])}")
        else:
            print(f"Error: {r.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_embeddings()
