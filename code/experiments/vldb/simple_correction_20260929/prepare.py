"""Copy the unchanged 172 frozen inputs into an independent evaluation run."""
from __future__ import annotations
import hashlib
import shutil
from collections import Counter
from .common import *


def copy_verified(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if sha(source) != expected:
        raise RuntimeError('Parent input changed: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != expected:
            raise RuntimeError('Existing copied input differs: ' + str(destination))
    else:
        temp = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
        shutil.copyfile(source, temp)
        if sha(temp) != expected:
            raise RuntimeError('Input copy failed SHA256 verification')
        temp.replace(destination)


def main():
    if (RUN / 'freeze.json').exists():
        value = frozen()
        for entry in value['entries']:
            arrays(entry)
        print({'entries': len(value['entries']), 'resumed': True}, flush=True)
        return
    if RUN.resolve() in (PARENT.resolve(), MAIN.resolve()):
        raise RuntimeError('A simple run must not overwrite a historical run')
    from experiments.vldb.baseline_expansion_20260922.common import PROTOCOL as parent_protocol
    parent_path = PARENT / 'freeze.json'
    parent = read(parent_path)
    if parent['protocol'] != parent_protocol:
        raise RuntimeError('Parent protocol differs from its frozen implementation')
    expected = {'forecast': 142, 'ucr': 6, 'anomaly': 24}
    if dict(Counter(e['task'] for e in parent['entries'])) != expected:
        raise RuntimeError('Unexpected parent input roster')
    entries_copy = []
    for entry in parent['entries']:
        entry = dict(entry)
        # Copy NPZ bytes, not np.savez: zip timestamps would alter input identity.
        source = PARENT / entry['input']
        target = RUN / entry['input']
        if not target.resolve().is_relative_to(RUN.resolve()):
            raise RuntimeError('Input path escapes new run')
        copy_verified(source, target, entry['input_sha256'])
        entries_copy.append(entry)
    upstream = dict(parent['upstream'])
    upstream[str(parent_path.relative_to(ROOT))] = sha(parent_path)
    for relative, expected_sha in upstream.items():
        if sha(ROOT / relative) != expected_sha:
            raise RuntimeError('Frozen upstream changed: ' + relative)
    record = dict(
        protocol=PROTOCOL,
        protocol_sha256=hashlib.sha256(canonical(PROTOCOL)).hexdigest(),
        created_utc=now(), entries=entries_copy, upstream=upstream,
        parent_binding=dict(run=str(PARENT.relative_to(ROOT)),
                            freeze_sha256=sha(parent_path),
                            protocol_sha256=parent['protocol_sha256'],
                            inputs='Exact NPZ byte copies; all 172 input SHA256 unchanged'),
        preparation_code={str(p.relative_to(ROOT)): sha(p) for p in
                          (Path(__file__), Path(__file__).with_name('common.py'))},
    )
    write(RUN / 'freeze.json', record, exclusive=True)
    print({'entries': len(entries_copy), **expected,
           'freeze_sha256': sha(RUN / 'freeze.json')}, flush=True)


if __name__ == '__main__':
    main()
