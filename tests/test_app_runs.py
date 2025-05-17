import subprocess
import sys
import time
import os

def test_streamlit_app_runs():
    # Start the Streamlit app in headless mode
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", os.path.abspath("app.py"), "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait a few seconds to see if it crashes
        time.sleep(10)
        # Check if the process is still running (should be)
        assert proc.poll() is None, "Streamlit app crashed on startup"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill() 