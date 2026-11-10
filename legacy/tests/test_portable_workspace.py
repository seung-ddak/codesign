import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import pytest
spec=importlib.util.spec_from_file_location('portable',Path(__file__).parents[1]/'code/portable_workspace.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def bundle(root, name='codesign/checkpoint.bin'):
    raw=b'checkpoint data';archive=root/'snapshot.part000'
    with tarfile.open(archive,'w:gz') as t:
        info=tarfile.TarInfo(name);info.size=len(raw);t.addfile(info,io.BytesIO(raw))
    manifest={'source_workspace':'/old/workspace','parts':[{'name':archive.name,'bytes':archive.stat().st_size,'sha256':p.digest(archive)}], 'files':{name:hashlib.sha256(raw).hexdigest()}}
    (root/'manifest.json').write_text(json.dumps(manifest));return archive


def test_restore_checksum_and_preserve_local_changes(tmp_path):
    source=tmp_path/'parts';source.mkdir();part=bundle(source);dest=tmp_path/'other_pc'
    p.restore(source,dest);p.restore(source,dest)
    (dest/'codesign/checkpoint.bin').write_bytes(b'local change')
    with pytest.raises(FileExistsError):p.restore(source,dest)
    part.write_bytes(b'corrupt')
    with pytest.raises(ValueError):p.verified_parts(source)


def test_reject_path_traversal(tmp_path):
    source=tmp_path/'parts';source.mkdir();bundle(source,'../escape')
    with pytest.raises(ValueError):p.restore(source,tmp_path/'new')


def test_rebase_absolute_paths_and_directory_symlinks(tmp_path):
    source=tmp_path/'parts';source.mkdir();bundle(source)
    workspace=tmp_path/'new';repo=workspace/'codesign';repo.mkdir(parents=True)
    report=repo/'report.html';report.write_text('file:///old/workspace/codesign/videos/result.mp4')
    link=repo/'references';link.symlink_to('/old/workspace/codesign/source',target_is_directory=True)
    p.rebase(source,workspace)
    assert str(workspace) in report.read_text() and '/old/workspace' not in report.read_text()
    assert link.readlink()==Path('source')
