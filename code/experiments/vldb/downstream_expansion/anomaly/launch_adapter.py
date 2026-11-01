"""Resume with the same cuDNN flags used by every original training job.

The frozen runner sets these flags during new training, but its load-checkpoint
branch returns before doing so. This startup-only adapter sets the identical
flags before either path, without changing models, weights, data, or metrics.
Added after initial compression/training launch and before any resume evaluation.
"""
import hashlib
from pathlib import Path
import torch
from . import run

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    identity = dict(adapter_sha256=run.sha(Path(__file__)),
        original_protocol_sha256=run.sha(run.OUT/'protocol.json'),
        cudnn_benchmark=False,cudnn_deterministic=True,
        reason='Restore the original new-training backend settings before loading existing checkpoints; no scientific-protocol change.')
    dest=run.OUT/'resume_runtime_adapter.json'
    if dest.exists():
        assert run.json.loads(dest.read_text())==identity
    else:
        run.write_json(dest,identity)
    run.main()
