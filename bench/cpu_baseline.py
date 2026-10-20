import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from checks.batch import main as batch_main
from checks.cache import main as cache_main


def triton_status():
    script = Path(__file__).with_name("triton_smoke.py")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if proc.returncode == 0:
        return {"ok": True}
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return {"ok": False, "error": tail[-1] if tail else f"exit {proc.returncode}"}


def main():
    cuda = torch.cuda.is_available()
    timings = {}
    timings.update(cache_main())
    timings.update(batch_main())
    report = {
        "milestone": "M0",
        "date": "2026-10-20",
        "host": "WSL2",
        "note": "CPU fp32 timings. The GPU is present and unused by these paths. Mac timings are not in this checkout.",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "nsight_systems": shutil.which("nsys") is not None,
        "nsight_compute": shutil.which("ncu") is not None,
        "triton_smoke": triton_status(),
        "mac_timings_s": None,
        "timings_s": timings,
        "command": "PYTHONPATH=. .venv/bin/python bench/cpu_baseline.py",
    }
    out = Path(__file__).resolve().parents[1] / "bench" / "results" / "2026-10-20-m0-cpu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
