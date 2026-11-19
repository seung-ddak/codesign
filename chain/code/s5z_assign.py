"""5z 단계: z 기준 selected.json (s4, 2026-11-17) 으로 s5_retarget/assignment.json 만 다시 만든다 (30 Hz 리타겟은 하지 않음; 100 Hz 리타겟 = s5h).
새로 뽑힌 CALM 클립만 a1_reference 로 변환 (s5_retarget.convert). 형상·레퍼런스 목록은 기존 assignment 그대로. 이전 파일 = assignment_speed.json."""
import json
from multiprocessing import Pool
from common import RUNS, log
from s5_retarget import OUT, convert, assign

if __name__ == '__main__':
    old = json.load(open(OUT / 'assignment.json')); sel = json.load(open(RUNS / 's4_calm/selected.json'))
    with Pool(16) as p: names = {ref: sorted(p.map(convert, files, chunksize=20)) for ref, files in sel.items()}
    _, by_group = assign(names, list(old['shapes']))
    json.dump(dict(shapes=old['shapes'], references=old['references'], by_group=by_group), open(OUT / 'assignment.json', 'w'), indent=1)
    log('s5z assignment (z 기준) ' + ', '.join(f'{g[4:-9]} {len(v)}' for g, v in names.items()))
