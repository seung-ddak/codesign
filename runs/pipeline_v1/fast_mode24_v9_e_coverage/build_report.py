"""v9 report: reads saved results only; one self-contained report.html."""
import base64, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

root = Path(__file__).resolve().parent
V8 = root.parent / "fast_mode24_v8_film"; V7 = root.parent / "fast_mode24_v7_eswap"
load = lambda p: json.loads(Path(p).read_text())
uri = lambda p, m: f"data:{m};base64," + base64.b64encode(Path(p).read_bytes()).decode()
test, train, scale = load(root / "given_a_test.json"), load(root / "given_a_train_shapes.json"), load(root / "true_scale.json")

def table(head, rows):
    return "<div class=scroll><table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>"

KO = dict(height="높이", tilt="기울기", linvel="속도", yaw="방향", pos="위치", joint="관절")
t = lambda g, n="v9", d=test: d[n]["start30"][g]
criteria = []
for g in ("height", "tilt", "linvel"):
    a, b = t(g, "A")["e_agree_moving"], t(g)["e_agree_moving"]
    pa, pb = t(g, "A")["wrongE_moving"] - t(g, "A")["pin_moving"], t(g)["wrongE_moving"] - t(g)["pin_moving"]
    criteria.append([f"{KO[g]} 움직임 일치도 +0.1", f"{a:.2f} → {b:.2f} ({b-a:+.2f})", "<b class=ok>통과</b>" if b - a >= .1 else "<b class=no>실패</b>"])
    criteria.append([f"{KO[g]} 틀린 E 벌점 증가", f"{pa:+.3f} → {pb:+.3f}", "<b class=ok>통과</b>" if pb - pa >= .005 else "<b class=no>실패</b> (0.005 미만, 사실상 동일)"])
CHECK = {"v7": V7 / "codesign_check", "v8 A": V8 / "A/check_w1", "v9": root / "check_w1"}
SMOKE = {"v7": V7 / "smoke_zigzag_joint_c16", "v8 A": V8 / "A/check_w1/smoke_zigzag_joint_c16", "v9": root / "check_w1/smoke_zigzag_joint_c16"}
gate, joint = [], []
for v, p in CHECK.items():
    cc = load(p / "codesign_check.json"); s = load(SMOKE[v] / "summary.json")["runs"]["scale_0"]
    e = float(np.median([r["path_err_m"] for r in s if r["path_err_m"] is not None and r["mjx_total"] is not None]))
    for k in ("zigzag", "circle"):
        fx = load(p / f"fixedE_{k}_4s/summary.json")["runs"]["scale_10"]
        pr = [np.mean([r["pred_total"] for r in fx[i*4:i*4+4]]) for i in range(16)]
        ac = [np.mean([r["mjx_total"] for r in fx[i*4:i*4+4] if r["mjx_total"] is not None]) for i in range(16)]
        rho = float(spearmanr(pr, ac).statistic); c = cc[k]; ok = c["pred_R"]["p"] < .05 and rho > .6 and e < .15
        gate.append([v, k, f'{c["pred_R"]["p"]:.3f}', f"{rho:+.2f}", f"{e:.2f} m", f'{c["pred_R"]["eta2"]:.2f}', f'{c["mjx_R"]["eta2"]:.2f}', "<b class=ok>통과</b>" if ok else "<b class=no>실패</b>"])
        j = c["joint"]; pe = np.median([r["path_err_m"] for r in fx if r.get("path_err_m") is not None])
        joint.append([v, k, f'{j["percentile_vs_designs"]:.0f}%', f'{j["median_R"]:.2f}', f'{c["fixed_median_R"]:.2f}', f'{j["median_progress"]:+.2f} m', f"{pe:.2f} m", f'{j["done"]}/16'])
        assert not ok
move_rows = [[KO[g]] + [f'{t(g, n)["e_agree_moving"]:.2f}' for n in ("v7", "A", "v9")] + [f'{t(g, "A", train)["e_agree_moving"]:.2f}', f'{t(g, "v9", train)["e_agree_moving"]:.2f}'] for g in KO]
pen_rows = [[KO[g], f'{t(g)["pin_moving"]:.3f}', f'{t(g)["wrongE_moving"]:.3f}', f'{t(g)["wrongA_moving"]:.3f}', f'{scale[g]["signal_moving"]:.3f}', f'{scale[g]["effect_moving"]:.3f}', f'{t(g)["pin_track"]:.2f}'] for g in KO]
static_rows = [[KO[g]] + [f'{t(g, n)["pin_rmse"]:.3f}' for n in ("v7", "A", "v9")] + [f'{t(g, n)["wrongE_rmse"]:.3f}' for n in ("v7", "A", "v9")] for g in ("height", "tilt", "pos", "linvel")]

def clip(k):
    folder = root / f"check_w1/joint_{k}_4s"; rows = load(folder / "summary.json")["runs"]["scale_10"]
    ok = [i for i, r in enumerate(rows) if r["mjx_total"] is not None]; out = ""
    for name, i in (("실제 보상 최고", max(ok, key=lambda x: rows[x]["mjx_total"])), ("실제 보상 최저", min(ok, key=lambda x: rows[x]["mjx_total"]))):
        r = rows[i]; p = folder / f"videos/scale_10_cand{i}.mp4"; fell = "4초 생존" if r["mjx_fell_step"] == -1 else f'{r["mjx_fell_step"]}스텝 낙상'
        out += f'<article><h4>{k} · {name}</h4><p>예측 R {r["pred_total"]:.2f} / 실제 R {r["mjx_total"]:.2f} / {fell}</p><video controls muted loop playsinline preload=metadata src="{uri(p, "video/mp4")}"></video></article>'
    return out
figs = "".join(f'<figure><figcaption>v9 · {k}: TT 예측 경로와 MJX 실제 경로</figcaption><img alt="{k} 경로" src="{uri(root / f"check_w1/joint_{k}_4s/paths.png", "image/png")}"></figure>' for k in ("zigzag", "circle"))
body = f"""
<h1>v9 · 형상 확대 결과</h1>
<p class=lead>학습 형상을 3배로 늘려도 처음 보는 형상에서 움직임 예측은 나아지지 않음. 문제는 E 하나가 아니라, 행동을 줘도 몸통 움직임을 예측하지 못하는 쪽으로 좁혀짐.</p>
<section class=verdict><h2>결론</h2><ul>
<li><b>판정: 실패.</b> 사전 기준 3개를 모두 통과하지 못함. 2초 경로오차는 0.12 m에서 {float(np.median([r["path_err_m"] for r in load(SMOKE["v9"] / "summary.json")["runs"]["scale_0"] if r["path_err_m"] is not None])):.2f} m로 악화됨.</li>
<li><b>문제:</b> 행동을 고정해도 TT의 몸통 움직임 오차가 움직임 크기 자체와 거의 같음. 높이 오차 {t("height")["pin_moving"]:.3f} / 크기 {scale["height"]["signal_moving"]:.3f}, 속도 {t("linvel")["pin_moving"]:.3f} / {scale["linvel"]["signal_moving"]:.3f}. 움직임을 "없다"고 예측하는 것과 거의 같은 수준임.</li>
<li><b>원인 판단:</b> 형상 다양성 부족은 주원인이 아님. 현재 학습 방식이 행동→몸통 움직임 관계 자체를 배우지 못한 것이 유력함. 이 관계가 없으면 E가 그 관계를 바꾸는 효과도 배울 수 없음.</li>
<li><b>다음:</b> 형상 추가는 멈춰야 함. 같은 데이터에 Mode 1(E·S0·A를 모두 주고 S만 예측)을 넣어 학습하는 진단이 먼저임.</li>
</ul></section>
<section><h2>검증 질문과 사전 기준</h2>
<div class=qa><p class=q>Q. 팔·레일 뺀 10좌표 안에서 형상을 채우면 E 움직임 일치도가 오르나?</p><p class=a>A. 오르지 않음. 데이터 부족 가설은 지지되지 않음.</p></div>
<div class=qa><p class=q>Q. 그럼 v8의 "움직임 일치도 ≈0"은 믿어도 되나?</p><p class=a>A. 그 측정은 결함이 있었음. TT가 행동까지 생성한 궤적과 비교했음. 이번엔 행동을 고정했고, 처음 보는 형상에서도 낮게 나옴. 결론은 유지되지만 v8 수치 자체는 쓰면 안 됨.</p></div>
<p>기준은 시험 형상 8개, 행동 고정, 시작 30, v8 A 대비임. 벌점 증가의 최소 폭 0.005는 사전 기준에 없어서 판정 표시를 위해 결과를 본 뒤 정했음. 벌점 차이는 모두 0.002 이하라 이 값이 판정을 좌우하지 않음.</p>
{table(["기준", "v8 A → v9", "판정"], criteria + [["기존 게이트", "p·순위상관·경로오차 모두 미달", "<b class=no>실패</b>"]])}</section>
<section><h2>고정 조건과 바꾼 변수</h2>
{table(["고정 조건", "값"], [["구조", "v8 A와 동일: E 전 층 주입, E 가리기 없음"], ["학습", "60k step, batch 32, horizon 32, Mode 2·4, diffusion 5단계, minmax, clip 1, DDIM"], ["검사", "고정 E 16형상 × 4시드, joint 16후보, zigzag·circle, scale 10, 4초 open-loop"]])}
{table(["변수", "v8 A", "v9"], [["학습 형상", "16개 (10좌표 중 독립 7방향)", "48개 (새 32개, Sobol, 독립 10방향)"], ["새 형상 데이터", "없음", "형상당 행동 128개, 100스텝, open-loop"], ["효과 없는 3좌표", "형상마다 다름", "동일하게 무작위"]])}
<p>시험 형상 8개(형상당 행동 512개)는 학습에서 뺐고, 시험 쌍은 검증용 행동 {test["v9"]["start30"]["pairs"]}쌍임. 모델 4개를 같은 쌍으로 비교함. 같은 60k step에 데이터가 늘어 형상당 학습량은 줄었음.</p></section>
<section><h2>결과 1. 게이트</h2>{table(["모델", "task", "형상 효과 p", "순위상관", "2초 경로오차", "예측 η²", "MJX η²", "판정"], gate)}
<p>검사 형상은 기존 16개임. v9의 예측 η²는 0.11·0.18로 오히려 낮음.</p></section>
<section><h2>결과 2. 움직임 일치도 (행동 고정, 시작 30)</h2>
{table(["항목", "시험 형상 v7", "시험 형상 v8 A", "시험 형상 v9", "학습 형상 v8 A", "학습 형상 v9"], move_rows)}
<p>v9는 시험 형상에서 기울기·방향·관절이 0.03~0.19 오르고 속도가 0.02 내림. 기준 항목 중 +0.1을 넘긴 것은 없음. 방향은 v8 A의 값이 −0.16으로 비정상적으로 낮아 상승 폭의 의미가 작음.</p></section>
<section><h2>결과 3. 핵심. 오차가 움직임 크기와 같음</h2>
<p>시험 형상, 시작 30, 정규화 단위. "움직임 크기"는 형상 하나의 실제 움직임, "실제 E 효과"는 같은 행동에서 두 형상의 움직임 차이임.</p>
{table(["항목", "맞는 E·A 오차", "틀린 E 오차", "틀린 A 오차", "움직임 크기", "실제 E 효과", "추종 상관"], pen_rows)}
<ul><li>실제 E 효과는 작지 않음. 움직임 크기와 비슷함. "E 효과가 원래 작아서 안 보인다"는 설명은 틀림.</li>
<li>몸통 항목은 맞는 입력을 줘도 오차가 움직임 크기와 비슷함. 틀린 E, 틀린 A를 넣어도 오차가 거의 안 늘어남.</li>
<li>관절만 예외임. 고정한 행동을 따라가고(상관 0.83), 틀린 A에서 오차가 0.14로 뜀. 관절은 행동 목표값과 거의 같아서 쉬운 항목임.</li></ul></section>
<section><h2>결과 4. 평균 자세는 조금 일반화됨</h2>
{table(["항목", "맞는 E v7", "v8 A", "v9", "틀린 E v7", "v8 A", "v9"], static_rows)}
<p>시험 형상의 높이 오차가 0.107 → 0.095로 줄었음. 형상 확대의 효과는 평균 자세 쪽에만 있음. 반면 학습 형상 오차는 소폭 나빠졌고, 최종 검증 손실은 v8 A 0.027 → v9 0.037임. 검증 집합이 달라 직접 비교는 안 됨.</p></section>
<section><h2>결과 5. joint 생성</h2>{table(["모델", "task", "joint 백분위", "joint R", "고정 E R", "joint 진행", "4초 고정 E 경로오차", "joint 생존"], joint)}<div class=grid>{figs}</div>
<h3>MJX 영상</h3><p>"생존"은 시간 내 생존이지 트랙 완주가 아님.</p><div class=grid>{clip("zigzag")}{clip("circle")}</div></section>
<section><h2>사실 / 추정 / 미검증</h2><h3>확인된 사실</h3><ul><li>게이트와 움직임 기준 모두 실패.</li><li>행동을 고정해도 몸통 움직임 오차 ≈ 움직임 크기.</li><li>실제 E 효과는 움직임 크기와 비슷하게 큼.</li><li>시험 형상의 평균 높이 오차는 개선됨.</li></ul>
<h3>원인 추정</h3><ul><li>Mode 2·4 학습은 S와 A를 함께 생성하므로 "주어진 A의 결과"를 직접 학습하지 않음.</li><li>시간 축 움직임은 평균 자세보다 손실에서 차지하는 비중이 작음.</li></ul>
<h3>미검증 가설</h3><ul><li>Mode 1을 넣으면 몸통 움직임 예측이 생김.</li><li>학습량을 늘리면 v9의 경로오차가 회복됨.</li><li>낙상 데이터 비중이 v9 품질을 낮췄음.</li></ul></section>
<section><h2>결론이 틀릴 수 있는 이유</h2><ul>
<li><b>행동 고정은 임시 방법:</b> 이 모델은 행동이 주어진 조건으로 학습되지 않았음. 생성 과정에서 행동을 덮어쓴 것이라 모델 능력을 낮게 잴 수 있음. 이것이 가장 큰 위험이고, Mode 1 진단이 이를 직접 해소함.</li>
<li><b>시드 평균:</b> 8시드 평균은 움직임을 줄여 오차를 키우거나 줄일 수 있음.</li>
<li><b>움직임 크기 표본:</b> 무작위 200행, 일치도·오차는 {test["v9"]["start30"]["pairs"]}쌍이라 표본이 다름. 비율은 대략값임.</li>
<li><b>시험 형상 8개:</b> 형상 수가 적고, 낙상이 잦은 형상(143/512, 219/512)이 있음. 살아남은 쌍만 쓰므로 쉬운 쪽으로 편향될 수 있음. 이번 비교에는 신뢰구간이 없음.</li>
<li><b>학습량 혼입:</b> 60k step 고정이라 형상 확대와 형상당 학습량 감소가 섞임.</li>
<li><b>모션 범위:</b> 새 형상도 원본 16형상의 행동만 실행함. 그 형상에 맞는 보행은 아님.</li></ul></section>
<section><h2>다음 분기</h2><p><b>파이프라인 경고:</b> Mode 1 진단은 Reward-agnostic 학습 범위 안이고 방향 이탈이 아님. 다만 게이트 판정용이 아니라 원인 분리용임.</p>
{table(["결과", "판단", "다음 행동"], [["A. Mode 1로 몸통 오차가 움직임 크기보다 크게 감소", "문제는 Mode 2·4 학습 방식", "Mode 1을 학습에 넣고 E 일치도와 게이트 재측정"], ["B. Mode 1도 오차 ≈ 크기", "모델·표현·손실 문제", "1스텝 예측 대조군으로 데이터에 학습 가능한 신호가 있는지 확인"], ["C. 몸통은 개선, E 벌점은 그대로", "E 전달 문제", "v9 데이터로 E 주입과 학습량 재검토"]])}
<p>추천 비교 기준: 시험 형상에서 몸통 움직임 오차가 움직임 크기의 70% 미만, 틀린 A 오차가 맞는 A 오차보다 1.2배 이상. 예상 약 1시간. 공유 PPO 문제 3은 이 진단 결과를 본 뒤 정해도 됨.</p></section>
<section><h2>재현</h2><pre>cd RL/codesign_cleanup
bash ../scripts/tmux_run.sh v9_coverage "bash runs/pipeline_v1/fast_mode24_v9_e_coverage/run.sh"
PYTHONPATH=code/tt_dynamics ../transformer-transformer/.venv/bin/python runs/pipeline_v1/fast_mode24_v9_e_coverage/true_scale.py
python3 runs/pipeline_v1/fast_mode24_v9_e_coverage/build_report.py</pre>
<p>코드: 브랜치 cleanup/shared-ppo, 커밋 525fec437 위. 로그: logs/tmux_run/v9_coverage_20261031_134536.log.</p></section>"""
style = (V8 / "build_report.py").read_text().split('style = """')[1].split('"""')[0]
(root / "report.html").write_text(f'<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>v9 형상 확대 결과</title><style>{style}</style><main>{body}</main></html>')
print((root / "report.html").stat().st_size // 1024, "KB")
for row in criteria + move_rows + pen_rows: print(row)
