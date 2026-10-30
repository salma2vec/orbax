"""Validate the native Grain/Orbax bridge on four pinned source combinations."""

import argparse
import hashlib
import importlib.metadata
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


def export(repo, revision, package, destination):
    archive = subprocess.check_output(['git', 'archive', revision, package], cwd=repo)
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(destination, filter='data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orbax-repo', type=Path, default=Path.cwd())
    parser.add_argument('--case-script', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revisions = {
        'orbax-base': 'b3df0a14578c75afbb00f60f56420b9f674fc93d',
        'orbax-fixed': 'eafa34001f9d20008b12915a426704f62d644121',
        'grain-base': 'd2adadec1b275251b098215abe5c6df8b16ed6da',
        'grain-fixed': '7980fba53631e1d88228b01e835af5d0c970299b',
    }
    wheel = importlib.metadata.distribution('grain')
    assert wheel.version == '0.2.18'
    results = []
    hashes = {}
    with tempfile.TemporaryDirectory(prefix='recovery-matrix-') as temporary:
        root = Path(temporary)
        grain_repo = root / 'grain-repo'
        subprocess.run([
            'git', 'clone', '--quiet', '--depth', '3', '--branch',
            'codex/flatmap-exhausted-checkpoint',
            'https://github.com/salma2vec/grain.git', str(grain_repo),
        ], check=True)
        for name, revision in revisions.items():
            package = 'grain' if name.startswith('grain') else 'checkpoint/orbax'
            repo = grain_repo if package == 'grain' else args.orbax_repo
            destination = root / name
            export(repo, revision, package, destination)
            if package == 'grain':
                protobuf = Path('grain/proto/execution_summary_pb2.py')
                shutil.copy2(wheel.locate_file(protobuf), destination / protobuf)
                native_dir = Path('grain/_src/python/experimental/index_shuffle/python')
                native_files = list(Path(wheel.locate_file(native_dir)).glob('index_shuffle_module*.so'))
                assert native_files, 'Grain wheel native prerequisite missing'
                for native_file in native_files:
                    shutil.copy2(native_file, destination / native_dir / native_file.name)
                changed_file = destination / 'grain/_src/python/dataset/transformations/flatmap.py'
            else:
                changed_file = destination / 'checkpoint/orbax/checkpoint/checkpoint_manager.py'
            hashes[name] = hashlib.sha256(changed_file.read_bytes()).hexdigest()

        for orbax, grain in itertools.product(['base', 'fixed'], repeat=2):
            name = f'orbax-{orbax}_grain-{grain}'
            grain_path = root / f'grain-{grain}'
            orbax_path = root / f'orbax-{orbax}' / 'checkpoint'
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            env['PYTHONPATH'] = os.pathsep.join(map(str, [grain_path, orbax_path]))
            completed = subprocess.run(
                [sys.executable, str(args.case_script.resolve())],
                env=env, text=True, capture_output=True, timeout=120,
            )
            (args.output / f'{name}.stdout').write_text(completed.stdout)
            (args.output / f'{name}.stderr').write_text(completed.stderr)
            assert completed.returncode == 0, f'{name}: {completed.stderr[-4000:]}'
            result = json.loads(completed.stdout.splitlines()[-1])
            assert Path(result['grain_source']).is_relative_to(grain_path)
            assert Path(result['orbax_source']).is_relative_to(orbax_path)
            assert result['history'] == ([0] if orbax == 'fixed' else [])
            assert result['records_after_restore'] == ([] if grain == 'fixed' else [2, 3])
            assert result['restored_via'] == ('latest' if orbax == 'fixed' else 'explicit_step_0_control')
            result.update(orbax=orbax, grain=grain)
            results.append(result)
            print(name, result['restored_via'], result['records_after_restore'], flush=True)
            (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    (args.output / 'receipt.json').write_text(json.dumps({
        'revisions': revisions, 'source_sha256': hashes, 'python': sys.version,
        'packages': {name: importlib.metadata.version(name) for name in
                     ['grain', 'array-record', 'cloudpickle', 'jax', 'jaxlib', 'numpy', 'tensorstore']},
        'scope': 'Single-process CPU, public Composite and Grain checkpoint handlers; synthetic records and model state.',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
