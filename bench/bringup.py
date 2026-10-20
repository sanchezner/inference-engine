import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def triton_status():
    script = Path(__file__).with_name("triton_smoke.py")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if proc.returncode == 0:
        return {"ok": True}
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return {"ok": False, "error": tail[-1] if tail else f"exit {proc.returncode}"}


def main():
    cuda = torch.cuda.is_available()
    report = {
        "date": "2026-10-20",
        "host": "WSL2",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "nsight_systems": shutil.which("nsys") is not None,
        "nsight_compute": shutil.which("ncu") is not None,
        "triton_smoke": triton_status(),
    }
    print(json.dumps(report, indent=2))
    if not cuda:
        raise SystemExit("cuda is not available")


if __name__ == "__main__":
    main()
