#!/usr/bin/env python3
import subprocess
import os
import webbrowser
import time
import sys


def launch_tensorboard():
    """Launch TensorBoard for monitoring PPO training"""

    # TensorBoard log directory (organized with other Eureka data)
    if os.name == "nt":  # Windows
        log_dir = os.path.expanduser(
            "~/AppData/LocalLow/DefaultCompany/AR4/EurekaRewards/tensorboard"
        )
    else:  # Linux/Mac
        log_dir = os.path.expanduser(
            "~/.config/unity3d/DefaultCompany/AR4/EurekaRewards/tensorboard"
        )

    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Port for TensorBoard
    port = 6006

    print(f"Launching TensorBoard on port {port}...")
    print(f"Log directory: {log_dir}")

    try:
        # Launch TensorBoard
        cmd = [
            sys.executable,
            "-m",
            "tensorboard",
            f"--logdir={log_dir}",
            f"--port={port}",
            "--reload_interval=5",
        ]
        process = subprocess.Popen(cmd)

        # Wait a bit for TensorBoard to start
        time.sleep(3)

        # Open browser
        url = f"http://localhost:{port}"
        print(f"\nOpening TensorBoard at {url}")
        print(
            "If the browser doesn't open automatically, navigate to the URL manually."
        )
        print("\nPress Ctrl+C to stop TensorBoard")

        webbrowser.open(url)

        # Keep the script running
        process.wait()

    except KeyboardInterrupt:
        print("\nStopping TensorBoard...")
        process.terminate()
        process.wait()
        print("TensorBoard stopped.")
    except Exception as e:
        print(f"Error launching TensorBoard: {e}")
        print("\nMake sure TensorBoard is installed:")
        print("  pip install tensorboard")


if __name__ == "__main__":
    launch_tensorboard()
