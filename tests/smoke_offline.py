"""Run the released codec's deterministic, data-free isolation test."""
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from experiments.vldb.framework_ablation_20261017.test_codec_isolation import run


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("passed") is not True:
        raise SystemExit(1)
