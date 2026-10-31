"""Start Phoenix, the FastAPI backend and the Gradio UI together."""

import subprocess
import time

import phoenix


def main():
    phoenix.launch_app()
    api_process = subprocess.Popen(["cc-copilot-api"])
    ui_process = subprocess.Popen(["cc-copilot-ui"])
    try:
        while api_process.poll() is None and ui_process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        api_process.terminate()
        ui_process.terminate()


if __name__ == "__main__":
    main()
