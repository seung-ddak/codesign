"""겹친 비교 영상: 같은 정책을 학습 설정 (A, 원래 색) 과 테스트 설정 (반투명 주황) 으로 실행해 한 장면에 그린다.
사용: overlay.py <테스트 설정 첫 글자들, 예: DEF> <형상들, 예: E0,E1> <모션들, 예: trotCCWSlow__r00,...>
출력: videos_overlay/<형상>_<모션>_A_vs_<설정>.mp4, results/overlay.json"""
import json, sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
import slip_eval as se
from render_tt import render_overlay
tests = sys.argv[1]; shapes = sys.argv[2].split(','); motions = sys.argv[3].split(',')
out = HERE / 'videos_overlay'; out.mkdir(exist_ok=True); res_f = HERE / 'results/overlay.json'
res = json.loads(res_f.read_text()) if res_f.exists() else []
cfg = {n[0]: (n, c) for n, c in se.CONFIGS.items()}
EN = {'A': 'A: pyramidal, impratio 1, iter 5 (training)', 'B': 'B: elliptic, impratio 10, iter 5', 'C': 'C: pyramidal, impratio 1, iter 50',
      'D': 'D: elliptic, impratio 10, iter 50', 'E': 'E: elliptic, impratio 30, iter 50', 'F': 'F: elliptic, impratio 100, iter 50'}
for K in shapes:
    for M in motions:
        ra, qa = se.run(K, M, cfg['A'][1], keep=True); ra['config'] = cfg['A'][0]; res.append(ra)
        for T in tests:
            rb, qb = se.run(K, M, cfg[T][1], keep=True); rb['config'] = cfg[T][0]; res.append(rb)
            st = lambda t: f"slip A {ra['slip_contact']:.3f} vs {T} {rb['slip_contact']:.3f} m/s   t = {t * .02:4.1f} s"
            render_overlay(qa, qb, se.SHAPES[K], out / f'{K}_{M}_A_vs_{T}.mp4', title=f"{K} · {M.replace('__r00', '')}",
                           sub=f"solid = A (training) / orange = {EN[T][3:]}", status_fn=st)
            print(K, M, T, ra['slip_contact'], rb['slip_contact'], flush=True)
            res_f.write_text(json.dumps(res, indent=1, ensure_ascii=False))
