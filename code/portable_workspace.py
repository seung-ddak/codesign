"""Restore the verified workspace snapshot and rebase local experiment paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified_parts(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    parts = []
    for row in manifest['parts']:
        path = folder / row['name']
        if path.parent != folder or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f'Invalid snapshot part: {path}')
        parts.append(path)
    return manifest, parts


def restore(folder, workspace):
    manifest, parts = verified_parts(folder)
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=workspace) as joined:
        for part in parts:
            with part.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    joined.write(block)
        joined.seek(0)
        with tarfile.open(fileobj=joined, mode='r:gz') as archive:
            members = archive.getmembers()
            for member in members:
                target = (workspace / member.name).resolve()
                if not target.is_relative_to(workspace.resolve()):
                    raise ValueError(f'Unsafe archive path: {member.name}')
                if member.isfile() and target.exists() and digest(target) != manifest['files'][member.name]:
                    raise FileExistsError(f'Local file differs; preserve it before restoring: {target}')
            archive.extractall(workspace, members=members, filter='data')
    for name, expected in manifest['files'].items():
        if digest(workspace / name) != expected:
            raise ValueError(f'Restored file checksum mismatch: {name}')
    print(f'Restored and verified {len(manifest["files"])} files in {workspace}')


def rebase(folder, workspace):
    old = json.loads((folder / 'manifest.json').read_text())['source_workspace']
    new = str(workspace.resolve())
    if old == new:
        print('Workspace path unchanged; no rebasing needed')
        return
    count = 0
    for base in [workspace / 'codesign', workspace / 'transformer-transformer']:
        for directory, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in {'.git', '.venv', 'jax_cache', '__pycache__', 'portability'}]
            for name in files + [d for d in dirs if (Path(directory) / d).is_symlink()]:
                path = Path(directory) / name
                if path.is_symlink():
                    target = os.readlink(path)
                    if target.startswith(old + '/'):
                        replacement = os.path.relpath(new + target[len(old):], path.parent)
                        path.unlink()
                        path.symlink_to(replacement)
                    continue
                if name == 'config.pkl':
                    # These are trusted project checkpoints, never load an untrusted pickle.
                    import pickle
                    from omegaconf import OmegaConf
                    config = pickle.loads(path.read_bytes())
                    if OmegaConf.is_config(config):
                        text = OmegaConf.to_yaml(config)
                        if old in text:
                            path.write_bytes(pickle.dumps(OmegaConf.create(text.replace(old, new))))
                            count += 1
                elif path.suffix in {'.json', '.yaml', '.yml', '.html', '.md', '.txt', '.csv', '.py', '.sh', '.log', '.xml', '.urdf'}:
                    try:
                        text = path.read_text()
                    except (UnicodeError, OSError):
                        continue
                    if old in text:
                        path.write_text(text.replace(old, new))
                        count += 1
    print(f'Rebased {count} text/config files from {old} to {new}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'restore', 'rebase'])
    parser.add_argument('--bundle', type=Path, default=ROOT / 'portability')
    parser.add_argument('--workspace', type=Path, default=ROOT.parent)
    args = parser.parse_args()
    if args.command == 'verify':
        manifest, _ = verified_parts(args.bundle.resolve())
        print(f'All {len(manifest["parts"])} parts verified')
    elif args.command == 'restore':
        restore(args.bundle.resolve(), args.workspace.resolve())
    else:
        rebase(args.bundle.resolve(), args.workspace.resolve())


if __name__ == '__main__':
    main()
