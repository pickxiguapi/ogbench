"""Write a deterministic SHA-256 manifest for a release artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', default='manifest.json')
    args = parser.parse_args()

    root = args.root.resolve()
    output = root / args.output
    temporary = output.with_suffix(output.suffix + '.tmp')
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path in (output, temporary):
            continue
        files.append(
            {
                'path': path.relative_to(root).as_posix(),
                'bytes': path.stat().st_size,
                'sha256': sha256(path),
            }
        )
        print(files[-1]['path'], flush=True)

    payload = {
        'format': 'leworldmodel_pp_artifact_manifest_v1',
        'files': files,
        'total_bytes': sum(item['bytes'] for item in files),
    }
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, output)
    print(f'Wrote {len(files)} files to {output}', flush=True)


if __name__ == '__main__':
    main()
