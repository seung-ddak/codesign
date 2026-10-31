"""Rebuild the <=90 MiB workspace snapshot before a cross-PC handoff."""
from pathlib import Path
import subprocess,os,tarfile,tempfile,hashlib,json,time
repo=Path(__file__).resolve().parents[1];root=repo.parent;out=repo/'portability';out.mkdir(exist_ok=True);selected=set();excluded=[]
def gitpaths(*args):
 return [repo/n for n in subprocess.check_output(['git','-C',str(repo),'ls-files',*args,'-z']).decode().split('\0') if n]
for p in gitpaths('--others','--ignored','--exclude-standard'):
 n=p.relative_to(repo).as_posix()
 if any(x in n.split('/') for x in ['.git','.venv','__pycache__','.pytest_cache','jax_cache','logs','tmp']) or n.startswith('outputs/') or n.endswith(('.pyc','.zip','.mjb','.wandb')) or '/files/media/' in n:continue
 if p.is_file() and not p.is_symlink():selected.add(p)
for parent in [root/'transformer-transformer',root/'external/a1-calm-main']:
 for directory,dirs,files in os.walk(parent):
  here=Path(directory)
  dirs[:]=[d for d in dirs if d not in {'.git','.venv','__pycache__','.pytest_cache','jax_cache','wandb','outputs','node_modules'} and not (here==root/'transformer-transformer' and (d=='checkpoints' or d.startswith('umi_on_legs_plus_plus_')))]
  for name in files:
   p=here/name
   if name.startswith('.env') or name.endswith(('.pyc','.zip','.wandb')) or name in {'.DS_Store','MUJOCO_LOG.TXT'}:continue
   if p.is_file() or p.is_symlink():selected.add(p)
# Large new files belong in the split snapshot, never as a single Git blob.
oversized=[]
for p in gitpaths('--others','--exclude-standard'):
 if p.is_file() and not p.is_symlink() and p.stat().st_size >= 90*1024**2 and not p.is_relative_to(out):
  selected.add(p);oversized.append(p.relative_to(repo).as_posix())
if oversized:
 with (repo/'.gitignore').open('a') as f:
  f.write('\n# Large originals restored from portability/manifest.json\n')
  for name in oversized:f.write('/'+name+'\n')
# Snapshot worktree content, including local changes, without Git internals or credentials.
files={};links={};raw=sum(p.stat().st_size for p in selected if not p.is_symlink());print('Selected',len(selected),'files,',round(raw/1024**2),'MiB',flush=True)
with tempfile.NamedTemporaryFile(prefix='codesign-workspace-',suffix='.tar.gz',dir=root,delete=False) as temp:archive=Path(temp.name)
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
 for p in sorted(selected):
  name=p.relative_to(root).as_posix()
  if p.is_symlink():
   target=p.resolve()
   if not target.is_relative_to(root):excluded.append(name);continue
   info=tar.gettarinfo(str(p),arcname=name);info.linkname=os.path.relpath(target,p.parent);tar.addfile(info);links[name]=info.linkname
  else:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(1024**2),b''):h.update(block)
   files[name]=h.hexdigest();tar.add(p,arcname=name,recursive=False)
parts=[]
with archive.open('rb') as src:
 i=0
 while block:=src.read(90*1024**2):
  p=out/f'workspace.tar.gz.part{i:03d}';p.write_bytes(block);parts.append({'name':p.name,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()});i+=1
archive.unlink()
manifest={'format':1,'source_workspace':str(root),'parts':parts,'files':files,'symlinks':links,'dependency_git':{'transformer-transformer':subprocess.check_output(['git','-C',str(root/'transformer-transformer'),'rev-parse','HEAD']).decode().strip()},'excluded':['Python virtualenv, Git internals, compiler caches, regenerated MJBs, temporary logs, duplicate ZIPs','Downloaded upstream transformer-transformer/checkpoints and umi_on_legs_plus_plus_* datasets (not used by current CALM shared PPO / codesign dynamics runs)'],'excluded_external_symlinks':excluded}
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('Packed',len(parts),'parts;',sum(x['bytes'] for x in parts),'bytes',flush=True)
