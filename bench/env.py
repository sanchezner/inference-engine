import platform
import shutil
import subprocess
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


_SMI_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "clocks.max.sm",
    "clocks.max.mem",
    "power.draw",
    "utilization.gpu",
)


def compute_apps() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout).strip()}
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return {"ok": True, "apps": lines}


def nvidia_smi() -> dict:
    query = ",".join(_SMI_FIELDS)
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout).strip()}
    values = [part.strip() for part in proc.stdout.strip().splitlines()[0].split(",")]
    if len(values) != len(_SMI_FIELDS):
        return {"ok": False, "error": proc.stdout.strip()}
    return {"ok": True, **dict(zip(_SMI_FIELDS, values))}


def snapshot_revision(repo: str) -> str:
    path = Path(hf_hub_download(repo, "config.json", local_files_only=True))
    parts = path.parts
    if "snapshots" not in parts:
        return "unknown"
    return parts[parts.index("snapshots") + 1]


def git_revision() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "sha": sha, "dirty": bool(dirty)}


def triton_version() -> str:
    try:
        import triton
    except Exception as exc:
        return f"import failed: {exc}"
    return triton.__version__


def environment() -> dict:
    cuda = torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0) if cuda else None
    return {
        "host": "WSL2",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton_version(),
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": props.name if props else None,
        "gpu_total_bytes": props.total_memory if props else None,
        "sm_count": props.multi_processor_count if props else None,
        "nsight_systems": shutil.which("nsys") is not None,
        "nsight_compute": shutil.which("ncu") is not None,
        "nvidia_smi": nvidia_smi(),
        "compute_apps": compute_apps(),
        "git": git_revision(),
    }
