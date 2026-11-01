"""Run the frozen main matrix and automatically write the final summary."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone

from .benchmark import DEFAULT_RUN, ROOT, write_json


def main() -> None:
    # O_EXCL prevents two background drivers writing to the same run.
    lock = DEFAULT_RUN / "DRIVER.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    record = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(DEFAULT_RUN / "DRIVER.json", record)
    process = subprocess.run([sys.executable, "-u", "-m", "experiments.vldb.main_benchmark.benchmark",
                              "run", "--output", str(DEFAULT_RUN)], cwd=ROOT, check=False)
    summary = subprocess.run([sys.executable, "-m", "experiments.vldb.main_results.summary",
                               "--run", str(DEFAULT_RUN)], cwd=ROOT, check=False)
    write_json(DEFAULT_RUN / "DRIVER_FINISHED.json", {**record,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_returncode": process.returncode, "summary_returncode": summary.returncode})


if __name__ == "__main__":
    main()
