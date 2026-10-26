"""Pipeline v1 최종 보고서 (2026-10-26): 문제 정의 + 구현 현황. 수치 출처 = remind.md 8-1~8-31, 그림은 run 결과에서 직접 계산."""
import base64, io, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).parent
V6 = here.parent / "fast_mode24_v6_steps60k"
css = (V6 / "report_template.html").read_text().split("<main>")[0].replace("Guidance Scale Sweep", "Pipeline v1 최종 보고서")
MUTED = "#6b7a82"


def fig_uri(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def style(a):
    a.grid(alpha=.25)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for el in [a.xaxis.label, a.yaxis.label, a.title, *a.get_xticklabels(), *a.get_yticklabels()]:
        el.set_color(MUTED)


# 그림 1: TT 정확도는 학습량으로 해결됐다 (remind 8-26, 8-27; 2 s, scale 0, 후보 16)
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
runs = ["v4\n3k step", "v5\n20k step", "v6\n60k step"]
ax[0].bar(runs, [0.48, 0.20, 0.05], color="#0f6f86"); ax[0].axhline(0.15, ls="--", lw=1, color="#1d7a46")
ax[0].set_title("path error TT vs MJX (m, 2 s)", fontsize=10)
ax[1].bar(runs[1:], [12, 16], color="#0f6f86"); ax[1].set_ylim(0, 16); ax[1].set_title("2 s survived / 16", fontsize=10)
for a in ax:
    style(a)
fig1 = fig_uri(fig)

# 그림 2: E 의 효과 — MJX 는 크고 TT 예측은 작다 (codesign_check.json)
cc = json.load(open(V6 / "codesign_check/codesign_check.json"))
fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(2)
ax.bar(x - .18, [cc[t]["mjx_R"]["eta2"] for t in ("zigzag", "circle")], .36, color="#0f6f86", label="MJX actual")
ax.bar(x + .18, [cc[t]["pred_R"]["eta2"] for t in ("zigzag", "circle")], .36, color="#c06a1b", label="TT predicted")
ax.set_xticks(x, ["zigzag", "circle"]); ax.set_ylim(0, 1); ax.set_title("variance of R explained by design E (eta²)", fontsize=10)
ax.legend(frameon=False, fontsize=8); style(ax)
fig2 = fig_uri(fig)

# 그림 3: 10 s open-loop 오차 누적 (remind 8-30, scale 0, 후보 8)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([2, 5, 10], [0.05, 0.32, 1.27], "o-", color="#c06a1b"); ax.set_xlabel("time (s)")
ax.set_title("path error TT vs MJX, open-loop, scale 0 (m)", fontsize=10); style(ax)
fig3 = fig_uri(fig)

STAGES = [  # 단계, 구현, 검증 상태, pill, 근거
    ("1. 리타겟", "A1 모션 25개 랜덤화 → topology 00000011 연속 형상 리타겟", "완료. 19 모션 × 17 형상 (학습 16 + held-out e003)", "ok", "retargeting/, 8-1"),
    ("2. 모션 expert PPO", "모션별 다형상 공유 expert (E 를 obs 로 받음)", "완료. wide16: 19 모션, 학습 형상 16", "ok", "experts_wide16/"),
    ("3. rollout 수집", "expert 를 MJX 에서 실행해 E·S·A 저장", "완료. 16 형상 × 304 episode = 4,864 ep, 유효 818,688 step", "ok", "wide16_action/rollouts/"),
    ("4. TT 학습", "Mode 2 (E·S0 → S·A) + Mode 4 (S0 위치·yaw → E·S·A), h32, 5 step DDIM", "2 s 정확도 통과 (v6, 경로 차 0.05 m)", "ok", "8-27"),
    ("5. reward guidance 생성", "E·S·A 공동 생성, 매 확산 step 에 미분 reward gradient", "실행됨. 전진을 만들지 못함 (MJX progress +0.07 m < 기준 +0.15)", "bad", "8-29, 8-30"),
    ("6. MJX open-loop 검증", "생성 E 로 모델 빌드, 생성 A 를 실행, 예측 S 와 비교", "구현 완료. 2 s 일치, 10 s 경로 차 1.27 m", "bad", "8-30"),
    ("7. co-design (E 선택)", "joint 생성 E 가 학습 형상보다 좋은지", "실패. TT 가 E 차이를 예측하지 못함", "bad", "8-31"),
]

stage_rows = "\n".join(f"<tr><td>{s}</td><td>{i}</td><td><span class='pill {p}'>{'통과' if p == 'ok' else '미달'}</span> {v}</td><td class='dim'>{r}</td></tr>"
                       for s, i, v, p, r in STAGES)

html = css + f"""<main>
<section>
  <div class="eyebrow">codesign · pipeline_v1 · 최종 보고서 · 2026-10-26</div>
  <h1>Pipeline v1: 문제는 무엇이고 어디까지 구현됐나</h1>
  <div class="lead">
  <p><b>결론: 파이프라인 7 단계는 모두 코드로 구현되어 끝까지 돈다. 2 초 구간에서는 TT 예측과 MJX 실제가 일치한다 (경로 차 0.05 m). 그러나 co-design 은 아직 성립하지 않는다. TT 가 형상 E 의 차이를 예측하지 못해서 reward guidance 가 좋은 E 를 고를 수 없다.</b></p>
  <p>남은 문제는 세 개다. (1) TT 의 E 민감도 부족 — 가장 근본적이다. (2) guidance 가 실제 전진을 만들지 못한다. (3) 10 초 open-loop 에서 예측 오차가 누적된다. (1) 이 해결되지 않으면 (2)·(3) 을 해결해도 E 선택은 개선되지 않는다. 다음 작업 = (1), E 반사실 데이터 추가 (설계 중).</p>
  </div>
</section>

<section>
  <h2>연구 질문</h2>
  <div class="lead">
  <p>4 족 로봇의 기본 동작 rollout (여러 형상) 만으로 dynamics 를 배운 TT 가, 학습하지 않은 복합 task (zigzag, 원형 트랙) 의 reward 로 guidance 받아 <b>형상 E · 상태 S · 행동 A 를 노이즈에서 함께 생성</b>하고, 생성한 E·A 를 MJX 에서 open-loop 로 실행했을 때 task 를 실제로 수행하는가.</p>
  <p>성공 기준: 생성 E·A 의 MJX 실행이 task 를 수행하고 (progress, 생존), TT 가 예측한 S 와 MJX 실제 S 가 일치하며, 생성 E 가 학습 형상보다 task 에 더 맞는다.</p>
  </div>
</section>

<section>
  <h2>고정 환경</h2>
  <div class="scroll"><table>
    <tr><td class="k">로봇</td><td>A1 topology 00000011, 연속 치수 E 13 차원 (질량·관성·충돌 형상 일관 변경), 토크 한계 ±33.5 N·m</td></tr>
    <tr><td class="k">시뮬레이터</td><td>MuJoCo MJX, 50 Hz 제어, A = 12 관절 목표각 (PD)</td></tr>
    <tr><td class="k">형상</td><td>학습 16 (e000~e016 중 e003 제외), held-out e003</td></tr>
    <tr><td class="k">데이터</td><td>wide16 expert 19 모션 rollout, 4,864 episode × 최대 260 step</td></tr>
    <tr><td class="k">TT</td><td>v6: Mode 2+4, horizon 32, 60000 step × batch 32, minmax 정규화, DDIM 5 step</td></tr>
    <tr><td class="k">task</td><td>zigzag (진폭 0.3 m, 주기 2 m, lane 반폭 0.15 m), circle (반지름 1 m, lane 반폭 0.3 m). R = progress − 20·lane − 20·fall − 0.05·action_rate</td></tr>
    <tr><td class="k">S0</td><td>E 로부터 계산한 home 자세 안착 상태 (S0 = g(E))</td></tr>
  </table></div>
</section>

<section>
  <h2>구현 현황</h2>
  <div class="scroll"><table>
    <tr><th>단계</th><th>구현 내용</th><th>상태</th><th>근거 (remind)</th></tr>
    {stage_rows}
  </table></div>
  <p class="note">코드 = code/tt_dynamics/ (train_tt.py, generate_task.py, joint_diffusion_guidance.py, task_reward.py, diag_*.py). 보조 옵션: --no-guide-s, --expert-warmup-s0, --fixed-e (co-design 대조군), --replan K (구현, 실험은 취소).</p>
</section>

<section>
  <h2>문제 1 (해결됨): TT 예측이 부정확했다</h2>
  <div class="lead">
  <p><b>원인 = 학습량 부족. 3000 → 60000 step 으로 늘려 2 s 경로 차 0.48 → 0.05 m, 완주 16/16, 예측-MJX R 차 0.05.</b></p>
  <p>배제한 원인: open-loop 자체 (D1, 1 s 위치 차 0.4~0.7 cm), 데이터 분포 밖 (D2, p99 초과 0.07 %), S0 분포 밖 (8-25, expert warm-up S0 로도 불변), 교차 재실행 action 데이터 부족 (8-26, 효과 없음).</p>
  </div>
  <div class="fig"><img src="{fig1}" alt="학습량별 경로 차와 완주" style="max-width:100%"></div>
</section>

<section>
  <h2>문제 2 (현재 최우선): TT 가 형상 E 를 무시한다</h2>
  <div class="lead">
  <p><b>형상 E 는 MJX 실제 R 분산의 86 % (zigzag) · 92 % (circle) 를 설명하는데, TT 예측 R 에서는 20 % · 15 % 이고 유의하지 않다 (p 0.70 · 0.91).</b> 형상별 평균의 예측-MJX 순위 상관은 +0.42 · +0.01. TT 는 거의 모든 형상에 R 0 ~ +1 을 예측하지만 MJX 는 −8 ~ +0.5 로 퍼진다.</p>
  <p>결과: joint guidance 가 고른 E 의 MJX R 은 학습 형상 16 개 대비 zigzag 25 · circle 81 백분위. 좋은 형상을 고르지 못한다.</p>
  <p>같은 증상은 이전부터 반복됐다: D4 (8-15, 같은 S0·A 에 E 만 교체 시 TT 전진·yaw 반응 기울기 0.01~0.27), D4 pairwise (8-21, yaw slope ≈ 0). 원인 가설: 데이터의 A 는 expert 가 E 를 보고 만든 행동 (A = π(S, E)) 이라, E 효과가 A 에 섞여 들어가 TT 가 E 를 따로 쓸 이유가 없다 (D5 에서 A 에 대해 같은 현상 확인).</p>
  </div>
  <div class="fig"><img src="{fig2}" alt="E 가 설명하는 분산: MJX 대 TT" style="max-width:100%"></div>
</section>

<section>
  <h2>문제 3: guidance 가 실제 전진을 만들지 못한다</h2>
  <div class="lead">
  <p><b>scale 0 → 10 에서 TT 예측 progress −0.11 → +0.23 m, MJX 실제 −0.07 → +0.07 m (기준 +0.15 미달).</b> guidance 가 TT 가 부정확한 영역으로 생성을 민다 (경로 차 0.05 → 0.11 m, MJX fall 벌점 증가를 TT 가 놓침).</p>
  <p>S 에 직접 gradient 를 끄면 (no-guide-s) 예측-실제는 다시 일치하지만 progress 도 사라진다 (MJX −0.02 m). reward 구성 문제는 아니다: 데이터의 실제 걷기 R +0.23~+0.68 > 정지 −0.03.</p>
  </div>
</section>

<section>
  <h2>문제 4: 10 초 open-loop 에서 오차가 누적된다</h2>
  <div class="lead">
  <p><b>guidance 없이도 경로 차 2 s 0.05 → 5 s 0.32 → 10 s 1.27 m.</b> S guidance scale 10 은 10 s 완주 1/8. window 단위 MJX 재계획 (--replan) 을 구현했으나 사용자 결정으로 실험은 취소하고 co-design 검사를 먼저 했다.</p>
  </div>
  <div class="fig"><img src="{fig3}" alt="10 초 open-loop 경로 차 누적" style="max-width:100%"></div>
</section>

<section>
  <h2>원인 4 분류 요약</h2>
  <div class="scroll"><table>
    <tr><th>분류</th><th>점검 결과</th></tr>
    <tr><td>버그</td><td>free 조건 E 마스크 (8-11), foot-floor 접촉 판정 등 audit 14 건 수정 (issue #6). 실행 parity 정상 (target_q 재생 오차 0). 현재 알려진 미해결 버그 없음</td></tr>
    <tr><td>reward</td><td>구성 문제 아님 (걷기 R &gt; 정지 R, action_rate 기여 ≤ 0.012)</td></tr>
    <tr><td>최적화</td><td>TT 학습량 부족 → 60k step 으로 해결. guidance 가 TT 부정확 영역으로 미는 문제 남음</td></tr>
    <tr><td>관측 (데이터)</td><td><b>E 효과가 expert 행동에 섞인 데이터 → TT 가 E 를 무시. 현재 최우선</b></td></tr>
  </table></div>
</section>

<section>
  <h2>다음 단계 (진행 중: 1번)</h2>
  <div class="lead">
  <p>E 반사실 데이터: 같은 행동 A 를 여러 형상 E 에 open-loop 로 실행해 "E 만 다를 때 결과가 어떻게 다른가" 를 TT 가 직접 보게 한다. 각 형상의 시작 상태는 그 형상의 home 자세 S0 = g(E) 를 쓴다.</p>
  <p>과거 시도와의 차이: Foot-Anchored v2 (8-23) 는 형상 사이에 같은 qpos 를 공유하려다 유효한 공통 시작이 없어 수집 단계에서 실패했다. v5_paired (8-26) 는 A 를 바꾼 반사실이라 E 와 무관했다. 이번에는 qpos 를 공유하지 않고 행동 A 만 공유한다.</p>
  <p>통과 기준 (안): co-design 검사 재실행 시 TT 예측 R 의 E η² 유의 (p &lt; 0.05), 형상 평균 예측-MJX 순위 상관 &gt; 0.6, 2 s 경로 차 &lt; 0.15 m 유지.</p>
  </div>
</section>

<section class="qa">
  <h2>질문과 답</h2>
  <div><p class="q">문제가 뭔지 지금 어디까지 구현이 되었는지에 대한 보고서를 작성해 final report를 새롭게 작성해달라는거야</p>
    <p class="a">이 문서다. 구현 = 7 단계 전부 동작. 문제 = TT 가 E 를 무시 (최우선), guidance 전진 부족, 10 s 오차 누적. TT 정확도 문제는 해결됨.</p></div>
  <div><p class="q">co-design 검사부터 진행해줘 과제는 지금처럼 지그재그랑 원형 돌기만 해볼게 재계획은 취소하자</p>
    <p class="a">문제 2 의 근거가 이 검사다. 상세 = fast_mode24_v6_steps60k/codesign_check/report.html.</p></div>
  <div><p class="q">지그재그랑 원형 트랙 도는거 다 영상 찍어서 보고서에 넣어주고 1번으로 진행해보자</p>
    <p class="a">영상은 co-design 검사 보고서에 넣는다 (렌더 재실행). 1번 = 위 "다음 단계".</p></div>
</section>

<section>
  <h2>파일 위치</h2>
  <div class="scroll"><table>
    <tr><td class="k">기록 정본</td><td>remind.md (8-1 ~ 8-31)</td></tr>
    <tr><td class="k">TT checkpoint</td><td>runs/pipeline_v1/fast_mode24_v6_steps60k/tt_joint</td></tr>
    <tr><td class="k">guidance sweep · 10 s</td><td>runs/pipeline_v1/fast_mode24_v6_steps60k/report.html</td></tr>
    <tr><td class="k">co-design 검사</td><td>runs/pipeline_v1/fast_mode24_v6_steps60k/codesign_check/report.html</td></tr>
    <tr><td class="k">학습량 비교</td><td>runs/pipeline_v1/fast_mode24_v5_compare/report.html</td></tr>
    <tr><td class="k">코드 audit</td><td>code/tt_dynamics/README_PIPELINE_AUDIT.md (issue #6)</td></tr>
  </table></div>
</section>
</main>
"""
(here / "report.html").write_text(html)
print(here / "report.html", f"{len(html) / 1e6:.2f} MB")
