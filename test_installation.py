#!/usr/bin/env python3
"""
Test script to validate Smart File AI Chatbot installation and connectivity
"""

import sys
import requests
import json
from pathlib import Path

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def test_python_version():
    print("✓ Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  Python {version.major}.{version.minor}.{version.micro} ✓")
        return True
    else:
        print(f"  ERROR: Python 3.9+ required, found {version.major}.{version.minor}")
        return False

def test_dependencies():
    print("✓ Testing required packages...")
    packages = {
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'chromadb': 'ChromaDB',
        'requests': 'Requests',
        'pydantic': 'Pydantic',
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  {name:15} ✓")
        except ImportError:
            print(f"  {name:15} ✗ MISSING")
            all_ok = False
    
    return all_ok

def test_ollama_connection(base_url="http://localhost:11434"):
    print("✓ Testing Ollama connection...")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m.get('name', 'Unknown') for m in data.get('models', [])]
            if models:
                print(f"  Ollama is running ✓")
                print(f"  Available models: {', '.join(models)}")
                return True
            else:
                print(f"  Ollama running but no models found")
                print(f"  Pull a model: ollama pull llama2")
                return False
        else:
            print(f"  Ollama API error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to Ollama at {base_url}")
        print(f"  Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False

def test_backend_connection(base_url="http://localhost:8000"):
    print("✓ Testing backend connection...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  Backend is running ✓")
            print(f"  Status: {data.get('status', 'unknown')}")
            print(f"  Model: {data.get('ollama', {}).get('model', 'unknown')}")
            print(f"  Documents in DB: {data.get('database', {}).get('total_documents', 0)}")
            return True
        else:
            print(f"  Backend error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  Backend not running at {base_url}")
        print(f"  Start it with: python backend/main.py")
        return False
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        return False

def test_file_structure():
    print("✓ Testing project structure...")
    required_dirs = ['backend', 'frontend', 'data', 'logs']
    required_files = {
        'backend': ['main.py', 'rag.py', 'llm.py', 'db.py', 'logger.py', 'file_processor.py'],
        'frontend': ['index.html', 'styles.css', 'script.js'],
    }
    
    all_ok = True
    for d in required_dirs:
        path = Path(d)
        if path.is_dir():
            print(f"  {d:15} ✓")
        else:
            print(f"  {d:15} ✗ MISSING")
            all_ok = False
    
    for location, files in required_files.items():
        for file in files:
            path = Path(location) / file
            if path.is_file():
                print(f"  {location}/{file:25} ✓")
            else:
                print(f"  {location}/{file:25} ✗ MISSING")
                all_ok = False
    
    return all_ok

def test_api_endpoints(base_url="http://localhost:8000"):
    print("✓ Testing API endpoints...")
    endpoints = [
        ('/health', 'GET'),
        ('/statistics', 'GET'),
        ('/models', 'GET'),
    ]
    
    all_ok = True
    for endpoint, method in endpoints:
        try:
            if method == 'GET':
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code in [200, 400, 500]:
                print(f"  {method:4} {endpoint:20} ✓")
            else:
                print(f"  {method:4} {endpoint:20} ✗ ({response.status_code})")
                all_ok = False
        except Exception as e:
            print(f"  {method:4} {endpoint:20} ✗ ({str(e)[:30]})")
            all_ok = False
    
    return all_ok

def main():
    print_header("Smart File AI Chatbot - Installation Test")
    
    results = {}
    
    print("1. ENVIRONMENT CHECK")
    print("-" * 60)
    results['python'] = test_python_version()
    results['dependencies'] = test_dependencies()
    
    print("\n2. EXTERNAL SERVICES")
    print("-" * 60)
    results['ollama'] = test_ollama_connection()
    
    print("\n3. BACKEND SERVICE")
    print("-" * 60)
    results['backend'] = test_backend_connection()
    
    print("\n4. PROJECT STRUCTURE")
    print("-" * 60)
    results['structure'] = test_file_structure()
    
    print("\n5. API ENDPOINTS")
    print("-" * 60)
    if results.get('backend'):
        results['api'] = test_api_endpoints()
    else:
        print("  Skipped (backend not running)")
        results['api'] = None
    
    print_header("TEST SUMMARY")
    
    for check, passed in results.items():
        if passed is None:
            status = "⊘ SKIPPED"
        elif passed:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"  {check:20} {status}")
    
    all_passed = all(v for v in results.values() if v is not None)
    
    print("\n" + "="*60)
    if all_passed:
        print("  ✓ ALL TESTS PASSED - System is ready!")
        print("  Open frontend at: frontend/index.html")
        print("="*60)
        return 0
    else:
        print("  ✗ SOME TESTS FAILED - See details above")
        print("  Common fixes:")
        print("  1. Ensure Ollama is running: ollama serve")
        print("  2. Start backend: python backend/main.py")
        print("  3. Pull a model: ollama pull llama2")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
