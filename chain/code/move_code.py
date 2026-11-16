"""실험 폴더의 코드 (*.py, *.sh) 를 <폴더>/code/ 로 옮기고 경로를 고친다 (2026-11-12 사용자 요청).
python: HERE = code 의 부모 (결과 경로 그대로), sys.path 의 HERE → CODE_DIR. sh: $S/<파일> → $S/code/<파일>, '$S' import 경로 → '$S/code'."""
import re, shutil, sys
from pathlib import Path
d = Path(sys.argv[1]).resolve(); code = d / 'code'; code.mkdir(exist_ok=True)
files = [f for f in d.iterdir() if f.suffix in ('.py', '.sh') and f.is_file()]
names = {f.name for f in files}
for f in files:
    s = f.read_text()
    if f.suffix == '.py':
        s = s.replace('HERE = Path(__file__).resolve().parent', 'CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent', 1)
        s = s.replace('sys.path.insert(0, str(HERE))', 'sys.path.insert(0, str(CODE_DIR))')
    else:
        for n in names: s = re.sub(r'(\$S|\$\{S\})/' + re.escape(n), r'\1/code/' + n, s)
        s = s.replace("sys.path.insert(0,'$S')", "sys.path.insert(0,'$S/code')")
    (code / f.name).write_text(s); f.unlink()
shutil.rmtree(d / '__pycache__', ignore_errors=True)
print(d.name, '→ code/:', sorted(names))
