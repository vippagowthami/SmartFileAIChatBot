import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "functiongemma:latest",
        "prompt": "Say hello world.",
        "stream": False
    }
    try:
        print(f"Connecting to {url}...")
        r = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            print(f"Response: {r.json().get('response')}")
        else:
            print(f"Error: {r.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_ollama()
