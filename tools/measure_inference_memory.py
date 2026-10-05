"""Fresh-process peak working set, including Python and native library imports."""
from pathlib import Path
import sys
import argparse
import ctypes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def memory():
    if sys.platform != "win32":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"peak_working_set_bytes": int(value if sys.platform == "darwin" else value * 1024)}
    # Microsoft PSAPI definition, retained in the artifact's methodology URL.
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [(field, ctypes.c_size_t) for field in [
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"peak_working_set_bytes": int(counters.PeakWorkingSetSize), "working_set_bytes": int(counters.WorkingSetSize),
            "peak_pagefile_bytes": int(counters.PeakPagefileUsage)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["norman", "wessels"])
    args = p.parse_args()
    initial = memory()
    import torch
    from threadpoolctl import threadpool_limits
    from opal.inference import Predictor
    from opal.artifacts import write_json, sha256
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    imported = memory()
    run = ROOT / f"results/runs/{args.dataset}-log2-primary"
    out = run / "inference_memory.json"
    if out.exists():
        raise FileExistsError("Retained memory measurement exists")
    model = Predictor(run / "checkpoint")
    targets = model.reference["targets"][:2]
    for _ in range(100):
        model.predict(*targets)
    write_json(out, {"before_import": initial, "after_import": imported, "after_load_and_100_predictions": memory(),
        "checkpoint_manifest_sha256": sha256(run / "checkpoint/manifest.json"), "threads": 1,
        "scope": "fresh CPU process, includes interpreter and native library overhead; not training or hosted service memory",
        "methodology_url": "https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters"})
    print(args.dataset, memory())


if __name__ == "__main__":
    main()
