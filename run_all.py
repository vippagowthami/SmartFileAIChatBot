import os
import sys
import subprocess
import platform
import webbrowser
import time


def find_venv_python():
    root = os.path.abspath(os.path.dirname(__file__))
    if platform.system() == "Windows":
        candidate = os.path.join(root, ".venv", "Scripts", "python.exe")
        if os.path.exists(candidate):
            return candidate
    else:
        candidate = os.path.join(root, ".venv", "bin", "python")
        if os.path.exists(candidate):
            return candidate
    return sys.executable


def start_process(cmd, cwd=None):
    return subprocess.Popen(cmd, cwd=cwd)


def main():
    python_exec = find_venv_python()
    repo_root = os.path.abspath(os.path.dirname(__file__))
    backend_dir = os.path.join(repo_root, "backend")
    frontend_dir = os.path.join(repo_root, "frontend")

    backend_cmd = [python_exec, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
    frontend_cmd = [python_exec, "-m", "http.server", "8080"]

    print("Starting backend (Uvicorn) in", backend_dir)
    backend_proc = start_process(backend_cmd, cwd=backend_dir)
    time.sleep(1)

    print("Starting frontend static server in", frontend_dir)
    frontend_proc = start_process(frontend_cmd, cwd=frontend_dir)

    # Give servers a moment to start
    time.sleep(2)
    url = "http://localhost:8080"
    print(f"Opening frontend at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        print("Please open your browser to", url)

    print("Servers started. Press Ctrl+C to stop both.")
    try:
        while True:
            time.sleep(1)
            # Poll children to see if they exited
            if backend_proc.poll() is not None:
                print("Backend process exited.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend server exited.")
                break
    except KeyboardInterrupt:
        print("Stopping servers...")
    finally:
        for p in (backend_proc, frontend_proc):
            try:
                if p and p.poll() is None:
                    p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
