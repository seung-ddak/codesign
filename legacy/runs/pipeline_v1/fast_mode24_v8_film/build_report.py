"""v8 report: reads saved results only; one self-contained report.html."""
import base64, html, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

root = Path(__file__).resolve().parent
V7 = root.parent / "fast_mode24_v7_eswap"
CHECK = dict(v7=V7 / "codesign_check", A=root / "A/check_w1", B=root / "B/check_w2")
SMOKE = dict(v7=V7 / "smoke_zigzag_joint_c16", A=root / "A/check_w1/smoke_zigzag_joint_c16", B=root / "B/check_w2/smoke_zigzag_joint_c16")
LABEL = dict(v7="v7 (기존)", A="A (E 전 층 주입)", B="B (A + E 가리기, 영향 2배)")
load = lambda p: json.loads(Path(p).read_text())
uri = lambda p, mime: f"data:{mime};base64," + base64.b64encode(Path(p).read_bytes()).decode()
sens = {k: load(root / f"sensitivity/{k}.json") for k in ("v7", "A", "B_w1", "B_w2")}
detail = load(root / "sensitivity/detail_start30.json")

def table(head, rows):
    return "<div class=scroll><table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>"

gate = {}
for v, path in CHECK.items():
    cc = load(path / "codesign_check.json")
    smoke = load(SMOKE[v] / "summary.json")["runs"]["scale_0"]
    err = float(np.median([r["path_err_m"] for r in smoke if r["path_err_m"] is not None and r["mjx_total"] is not None]))
    gate[v] = dict(smoke=err, tasks={})
    for t in ("zigzag", "circle"):
        fx = load(path / f"fixedE_{t}_4s/summary.json")["runs"]["scale_10"]
        assert len(fx) == 64
        pr = [np.mean([r["pred_total"] for r in fx[i*4:i*4+4]]) for i in range(16)]
        ac = [np.mean([r["mjx_total"] for r in fx[i*4:i*4+4] if r["mjx_total"] is not None]) for i in range(16)]
        rho = float(spearmanr(pr, ac).statistic)
        gate[v]["tasks"][t] = dict(cc=cc[t], rho=rho, ok=cc[t]["pred_R"]["p"] < .05 and rho > .6 and err < .15)
assert not any(x["ok"] for g in gate.values() for x in g["tasks"].values())

mark = lambda ok: '<b class=ok>통과</b>' if ok else '<b class=no>실패</b>'
gate_rows = []
for v in CHECK:
    for t in ("zigzag", "circle"):
        x = gate[v]["tasks"][t]; c = x["cc"]
        gate_rows.append([LABEL[v], t, f'{c["pred_R"]["p"]:.3f}', f'{x["rho"]:+.2f}', f'{gate[v]["smoke"]:.2f} m', f'{c["pred_R"]["eta2"]:.2f}', f'{c["mjx_R"]["eta2"]:.2f}', mark(x["ok"])])
joint_rows = []
for v in CHECK:
    for t in ("zigzag", "circle"):
        c = gate[v]["tasks"][t]["cc"]; j = c["joint"]
        joint_rows.append([LABEL[v], t, f'{j["percentile_vs_designs"]:.0f}%', f'{j["median_R"]:.2f}', f'{c["fixed_median_R"]:.2f}', f'{j["median_progress"]:+.2f} m', f'{c["fixed_median_progress"]:+.2f} m', f'{j["done"]}/16'])

KO = dict(height="높이", tilt="기울기", joint="관절", yaw="방향", pos="위치", linvel="속도")
split_rows = []
for g in KO:
    r = [KO[g]]
    for n in ("v7", "A", "B_w2"):
        d = detail[n]["E_only"][g]
        r += [f'{d["all"]:.2f}<small> [{d["ci"][0]:.2f}, {d["ci"][1]:.2f}]</small>', f'{d["static"]:.2f}', f'{d["moving"]:+.2f}']
    split_rows.append(r)
size_rows = [[KO[g]] + [f'{sens[n]["start0"]["E_only"][g]["size_ratio"]:.2f}' for n in sens] + [f'{sens[n]["start0"]["E_only"][g]["agreement"]:.2f}' for n in sens] for g in ("height", "joint", "tilt", "yaw")]
tr = lambda n: {"v7": "v7", "A": "A", "B_w1": "B 영향 1배", "B_w2": "B 영향 2배"}[n]

def clip(v, t):
    folder = CHECK[v] / f"joint_{t}_4s"
    rows = load(folder / "summary.json")["runs"]["scale_10"]
    ok = [i for i, r in enumerate(rows) if r["mjx_total"] is not None]
    i = max(ok, key=lambda k: rows[k]["mjx_total"]); w = min(ok, key=lambda k: rows[k]["mjx_total"])
    out = []
    for name, k in (("실제 보상 최고", i), ("실제 보상 최저", w)):
        p = folder / f"videos/scale_10_cand{k}.mp4"; r = rows[k]
        media = f'<video controls muted loop playsinline preload=metadata src="{uri(p, "video/mp4")}"></video>' if p.exists() else "<p>영상 없음</p>"
        fell = "4초 생존" if r["mjx_fell_step"] == -1 else f'{r["mjx_fell_step"]}스텝 낙상'
        out.append(f'<article><h4>{LABEL[v]} · {t} · {name}</h4><p>예측 R {r["pred_total"]:.2f} / 실제 R {r["mjx_total"]:.2f} / {fell}</p>{media}</article>')
    return "".join(out)

figs = "".join(f'<figure><figcaption>{LABEL[v]} · {t}: TT 예측 경로와 MJX 실제 경로</figcaption><img alt="{v} {t} 경로" src="{uri(CHECK[v] / f"joint_{t}_4s/paths.png", "image/png")}"></figure>' for v in ("A", "B") for t in ("zigzag", "circle"))
videos = "".join(clip(v, t) for v in ("A", "B") for t in ("zigzag", "circle"))

body = f"""
<h1>v8 · E 주입 강화 결과</h1>
<p class=lead>TT에 E를 더 세게 넣어도 형상 선택 게이트는 통과하지 못함. 평균 자세 예측은 좋아졌지만 움직임 변화 예측은 그대로임.</p>

<section class=verdict><h2>정정 (v9에서 확인)</h2><p>아래 "움직임 일치도"는 TT가 행동까지 생성한 궤적과 같은 행동의 실제 결과를 비교한 값이라 측정 결함이 있음. 행동 고정 재측정과 후속 판단은 <code>fast_mode24_v9_e_coverage/report.html</code>을 봐야 함. 형상 추가 권고도 v9에서 지지되지 않았음.</p></section>
<section class=verdict><h2>결론</h2>
<ul>
<li><b>판정: 실패.</b> A·B 모두 zigzag·circle 게이트 4건을 전부 통과하지 못함.</li>
<li><b>문제:</b> E를 바꿔도 예측 움직임의 시간 변화가 실제와 맞지 않음. 일치도는 높이 −0.04~−0.01, 기울기 0.02~0.12, 관절 0.06~0.09, 속도 0.01~0.04임.</li>
<li><b>원인 판단:</b> 구조 문제가 주원인일 가능성은 낮아짐. 같은 데이터에서는 어떤 구조로도 E의 동적 효과를 학습하기 어렵다는 쪽이 유력함. 단, 데이터 원인은 아직 추정임.</li>
<li><b>이번 측정이 바로잡은 것:</b> 기존 "E 교체 일치도 ≈0"은 일부 틀렸음. 실제 생성 궤적으로 재면 v7 높이도 0.86임. 약한 것은 E 전체가 아니라 움직임 쪽 영향임.</li>
<li><b>다음:</b> E 주입 추가 개선은 멈춰야 함. 같은 10좌표 안에서 형상 수를 늘려 빈 방향을 채우는 데이터 실험으로 가야 함. 현재 16형상은 독립 방향 7개만 사용함.</li>
</ul></section>

<section><h2>검증 질문과 사전 기준</h2>
<div class=qa><p class=q>Q. 데이터는 그대로 두고 TT 구조만 바꾸면 E를 제대로 읽나?</p><p class=a>A. 평균 자세는 조금 나아졌지만 움직임 변화와 형상 순위는 나아지지 않았음.</p></div>
<div class=qa><p class=q>Q. E 영향을 2배로 키우면 되나?</p><p class=a>A. 안 됨. 예측 변화 크기만 커지고 일치도는 그대로임. 방향이 틀린 신호가 같이 커짐.</p></div>
{table(["기준", "값", "결과"], [
 ["E 교체 일치도 (높이)", "0.3 이상", "A 0.86 / B 0.83. 통과지만 v7도 0.86이라 개선 근거가 아님"],
 ["E 교체 일치도 (관절)", "0.3 이상", "A 0.14 / B 0.12. <b class=no>실패</b>"],
 ["예측 보상의 형상 효과", "p &lt; 0.05", "최저 A zigzag 0.14. <b class=no>실패</b>"],
 ["예측·실제 형상 순위상관", "&gt; 0.6", "최고 A zigzag +0.49. <b class=no>실패</b>"],
 ["2초 경로오차 중앙값", "&lt; 0.15 m", "A 0.12 / B 0.12. 통과"]])}
<p>일치도 기준은 측정 방법을 바꾸기 전에 정한 값임. 위 수치는 새 측정 기준이고, 결과를 본 뒤 기준값을 바꾸지 않았음.</p></section>

<section><h2>고정 조건과 바꾼 변수</h2>
{table(["고정 조건", "값"], [
 ["데이터", "v7과 동일: wide16_action + e_swap full (16형상 × 32 action × 16 대상 형상)"],
 ["학습", "60k step, batch 32, horizon 32, Mode 2·4, diffusion 5단계, minmax, clip 1, DDIM"],
 ["검사", "고정 E 16형상 × 4시드, joint 16후보, zigzag·circle, scale 10, 4초 open-loop, 재계획 없음"],
 ["smoke", "2초, 후보 16, scale 0, zigzag"],
 ["일치도 측정", "E 교체 데이터의 검증용 쌍 시작 0: 433쌍, 시작 30: 382쌍 (43그룹), 공통 시드 8개"]])}
{table(["변수", "v7", "A", "B"], [
 ["E 전 층 주입", "없음", "있음", "있음"],
 ["학습 중 E 가리기", "없음", "없음", "15%"],
 ["평가 시 E 영향", "1배", "1배", "2배 (1배도 민감도만 측정)"]])}
<p>B는 두 변수를 함께 바꿨음. A 대비 차이는 분리되지 않음.</p></section>

<section><h2>결과 1. 게이트</h2>{table(["모델", "task", "형상 효과 p", "순위상관", "2초 경로오차", "예측 η²", "MJX η²", "판정"], gate_rows)}
<p>MJX η²는 0.71~0.88로 실제 물리에서는 형상 차이가 큼. TT 예측은 0.12~0.34에 머물렀고, 순위상관은 task·모델마다 부호까지 바뀌어 안정적인 개선이 없음.</p></section>

<section><h2>결과 2. E가 평균에는 듣고 움직임에는 안 들음</h2>
<p>시작 30 기준, E만 교체했을 때의 실제 변화 대비 예측 변화 일치도임. 전체 = 전 구간, 평균 = 쌍별 시간 평균, 움직임 = 시간 평균을 뺀 나머지. 괄호는 그룹 부트스트랩 95% 구간임.</p>
{table(["항목", "v7 전체", "v7 평균", "v7 움직임", "A 전체", "A 평균", "A 움직임", "B(2배) 전체", "B(2배) 평균", "B(2배) 움직임"], split_rows)}
<p>A는 관절 전체 일치도를 0.06 → 0.16으로 올렸고, 기울기도 0.45 → 0.54로 올렸음. 관절은 구간이 겹치지 않아 잡음이 아니고, 기울기는 구간 경계가 닿음. 그러나 늘어난 부분은 평균 쪽이고, 움직임은 0.06 → 0.09에 그침. 위치만 움직임 일치도 0.35~0.38을 유지하지만 v7과 차이가 없음.</p></section>

<section><h2>결과 3. E 영향 2배의 효과</h2>
<p>시작 0 기준. 크기 비율 = 예측 변화 크기 / 실제 변화 크기.</p>
{table(["항목"] + [f"크기 {tr(n)}" for n in sens] + [f"일치도 {tr(n)}" for n in sens], size_rows)}
<p>B의 크기 비율은 0.42~0.45에서 0.75~0.83으로 커졌지만 일치도는 변하지 않음. 강도 조절로 풀리는 문제가 아님.</p></section>

<section><h2>결과 4. 학습과 joint 생성</h2>
<p>최종 검증 손실은 v7 0.0277, A 0.0265, B 0.0267임. 손실은 소폭 줄었고, A·B 모두 E 교체 데이터 오차가 원본 데이터보다 큼 (Mode 2: 0.0141 vs 0.0087, Mode 4: 0.047 vs 0.037).</p>
{table(["모델", "task", "joint 백분위", "joint R", "고정 E R", "joint 진행", "고정 E 진행", "joint 생존"], joint_rows)}
<p>joint 백분위는 v7 44·50에서 A 31·38, B 38·63임. B circle만 우세하고 나머지는 하락임. 후보 16개라 이 한 값만으로 개선이라 할 수 없음. 2초 경로오차는 0.22 → 0.12 m로 줄었지만 4초 고정 E 경로오차는 v7 0.40·0.41, A 0.57·0.75, B 0.41·0.61로 줄지 않음.</p>
<div class=grid>{figs}</div></section>

<section><h2>결과 5. 영상</h2><p>모두 MJX 실제 실행임. "done"은 시간 내 생존이지 트랙 완주가 아님.</p><div class=grid>{videos}</div></section>

<section><h2>사실 / 추정 / 미검증</h2>
<h3>확인된 사실</h3><ul>
<li>A·B 모두 게이트 실패, MJX 형상 효과는 여전히 큼.</li>
<li>E 교체의 평균 일치도는 높고(높이 0.91~0.93) 움직임 일치도는 낮음(−0.04).</li>
<li>E 영향을 2배로 키워도 일치도는 그대로임.</li>
<li>A의 관절 일치도 상승은 구간상 잡음이 아님.</li></ul>
<h3>원인 추정</h3><ul>
<li>16형상으로는 E→움직임 관계를 일반화하기 부족함.</li>
<li>TT가 시작 상태에서 형상 정보를 이미 얻을 수 있어 E 입력의 추가 정보가 적음 (안착 자세로 형상이 100% 식별됨).</li></ul>
<h3>미검증 가설</h3><ul>
<li>10좌표의 빈 방향을 채우는 형상 추가가 움직임 일치도를 올림.</li>
<li>A를 E에 다시 묶지 않은 교차 재실행이 필요함.</li>
<li>60k보다 긴 학습이 도움됨. 현재 근거로는 v8에서 재검증하지 않았음.</li></ul></section>

<section><h2>결론이 틀릴 수 있는 이유</h2><ul>
<li><b>S0 혼입:</b> E 교체 데이터는 형상마다 시작 상태가 달라, "E만 교체" 비교의 실제 변화에도 S0 차이가 섞임. 시작 30에서도 두 쌍의 상태가 달라 분리되지 않음. 같은 S0로 E만 바꾼 MJX 정답은 없음.</li>
<li><b>평균 후 비교:</b> 시드 8개 평균이라 생성 궤적의 움직임 진폭이 줄어듦. 움직임 일치도가 낮게 나오는 방향으로 편향될 수 있음.</li>
<li><b>일치도 잡음 바닥 미측정:</b> 데이터 자체의 재현 한계나 정답 입력에서의 최댓값을 재지 않았음. 움직임 일치도의 상한이 얼마인지 모름.</li>
<li><b>검증 그룹 수:</b> 43그룹이고 쌍을 그룹당 12개 복원 추출함. 부트스트랩은 그룹 단위로 했지만 시작 30 결과에만 적용함.</li>
<li><b>게이트 표본:</b> 형상당 4시드, joint 16후보라 순위상관·백분위 변동이 큼. v7 zigzag −0.22 → A +0.49 같은 변화는 단일 측정임.</li>
<li><b>측정 변경:</b> 기존 단일 단계 측정과 이번 생성 궤적 측정은 숫자를 직접 비교할 수 없음. 왜 달라졌는지도 확인하지 않았음.</li></ul></section>

<section><h2>다음 분기</h2>
<p><b>파이프라인 경고:</b> 모델 구조가 바뀌어 v8 체크포인트는 옛 모델과 호환되지 않음. 현재 코드는 미커밋임. 6단계 방향 이탈은 없음.</p>
{table(["결과", "판단", "다음 행동"], [
 ["A. 형상 추가 데이터에서 움직임 일치도 상승", "데이터 부족 확정", "형상 수를 늘려 게이트 재검사"],
 ["B. 형상을 늘려도 0 근처 유지", "학습 목표 또는 측정 문제", "일치도 상한부터 측정한 뒤 시작 상태 조건을 재검토"],
 ["C. 평균만 더 오르고 게이트 실패", "E→정적 자세만 학습", "E별 보행 차이가 데이터에 충분한지 PPO 수집부터 재검토"]])}
<p>추천: A를 기본 구조로 유지하고 B의 E 가리기는 쓰지 않아야 함. B는 A보다 나은 근거가 없음. 먼저 공유 PPO 문제 3(near_short 학습 조건) 결정이 필요함.</p></section>

<section><h2>재현</h2><pre>cd RL/codesign_cleanup
bash ../scripts/tmux_run.sh v8_film "bash runs/pipeline_v1/fast_mode24_v8_film/run.sh"
../transformer-transformer/.venv/bin/python runs/pipeline_v1/fast_mode24_v8_film/e_sensitivity_detail.py
python3 runs/pipeline_v1/fast_mode24_v8_film/build_report.py</pre>
<p>코드: 브랜치 cleanup/shared-ppo, 기준 커밋 25f613ba5 + 미커밋 변경 / TT 49f1fe1 + 미커밋 한 줄. 로그: logs/tmux_run/v8_film_20261031_082558.log, v8_detail_20261031_122937.log. 결과: A/, B/, sensitivity/.</p></section>
"""
style = """:root{--bg:#fff;--fg:#1c2733;--mut:#5a6875;--line:#d8e0e6;--card:#f5f8fa;--ok:#14733f;--no:#b42318;--acc:#0b5f83}
@media (prefers-color-scheme:dark){:root{--bg:#10171d;--fg:#e4edf3;--mut:#9fb0bc;--line:#33434f;--card:#18222a;--ok:#5fd08f;--no:#ff8f83;--acc:#7cc9ea}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,'Noto Sans KR',sans-serif}main{max-width:1120px;margin:auto;padding:24px 16px 64px}
h1{font-size:30px;margin:.2em 0}.lead{font-size:18px;color:var(--mut)}section{margin:32px 0;padding-top:8px;border-top:1px solid var(--line)}.verdict{border:0;background:var(--card);padding:8px 20px;border-radius:8px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}th,td{padding:7px 9px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}td:first-child,td:last-child{white-space:normal}th{color:var(--mut)}small{color:var(--mut)}
.ok{color:var(--ok)}.no{color:var(--no)}.qa{background:var(--card);padding:10px 16px;border-radius:8px;margin:10px 0}.qa p{margin:4px 0}.q{font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:16px}figure,article{margin:0;background:var(--card);padding:12px;border-radius:8px}h4{margin:0 0 4px}img,video{width:100%;display:block;margin-top:8px}figcaption{color:var(--mut);font-size:14px}pre{overflow-x:auto;background:var(--card);padding:12px;border-radius:8px;font-size:13px}"""
(root / "report.html").write_text(f'<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>v8 E 주입 결과</title><style>{style}</style><main>{body}</main></html>')
print(root / "report.html", (root / "report.html").stat().st_size // 1024, "KB")
