from __future__ import annotations

import subprocess
import sys
import time
import locale
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "followup_overnight"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_step(name: str, cmd: list[str], log_file) -> None:
    started = datetime.now()
    log_file.write(f"\n=== {name} started {started:%Y-%m-%d %H:%M:%S} ===\n")
    log_file.write(f"command: {' '.join(cmd)}\n")
    log_file.flush()

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
    code = proc.wait()
    dt = time.perf_counter() - t0
    log_file.write(f"=== {name} finished code={code} elapsed={dt:.1f}s ===\n")
    log_file.flush()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUT_DIR / f"followup_run_{stamp}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"root: {ROOT}\n")
        log_file.write(f"python: {sys.executable}\n")
        log_file.write(f"log: {log_path}\n")
        log_file.flush()

        run_step(
            "D-family model-performance analysis",
            [sys.executable, "scripts/analyze_d_family_model_performance.py"],
            log_file,
        )
        run_step(
            "C2 signature statistical analysis",
            [sys.executable, "scripts/eval_c2_signature_statistics.py", "--bootstrap", "5000"],
            log_file,
        )

        log_file.write("\nAll follow-up experiments finished successfully.\n")
    print(f"[done] log saved at {log_path}")


if __name__ == "__main__":
    main()

