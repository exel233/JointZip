"""Verify released file hashes and provenance. Does not run experiments."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_entry(path, digest, size=None):
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT) or not target.is_file():
        raise ValueError(f'Missing or out-of-tree file: {path}')
    if sha(target) != digest:
        raise ValueError(f'Hash mismatch: {path}')
    if size is not None and target.stat().st_size != size:
        raise ValueError(f'Size mismatch: {path}')


def main():
    count = 0
    for line in (ROOT / 'MANIFEST.sha256').read_text(encoding='utf-8-sig').splitlines():
        digest, path = line.split('  ', 1)
        check_entry(path, digest)
        count += 1
    provenance = {}
    for name in ['code/SOURCE_MANIFEST.json', 'results/SOURCE_MANIFEST.json', 'paper/SOURCE_MANIFEST.json']:
        entries = json.loads((ROOT / name).read_text(encoding='utf-8-sig'))
        if isinstance(entries, dict):
            entries = entries['files']
        for row in entries:
            check_entry(row.get('file', row.get('path')), row['sha256'], row['bytes'])
        provenance[name] = len(entries)
    # Only package documentation: frozen scientific README references can point
    # to upstream artifacts which are intentionally acquired separately.
    links = 0
    for doc in [ROOT / 'README.md', *sorted((ROOT / 'docs').glob('*.md'))]:
        for dest in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', doc.read_text(encoding='utf-8-sig')):
            if '://' in dest or dest.startswith(('#', 'mailto:')):
                continue
            dest = dest.split('#', 1)[0]
            if not (doc.parent / dest).exists():
                raise ValueError(f'Broken local link in {doc.relative_to(ROOT)}: {dest}')
            links += 1
    print(json.dumps({'passed': True, 'manifest_files_checked': count,
                      'provenance_files_checked': provenance, 'local_links_checked': links,
                      'scientific_rerun': False}, indent=2))


if __name__ == '__main__':
    main()
