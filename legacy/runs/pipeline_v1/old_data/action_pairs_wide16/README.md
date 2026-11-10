# action_pairs_wide16: A 쌍 데이터 (같은 s0·E, 다른 A) 16 형상

- 용도: remind 8-49 Δ-loss 의 A 쌍. 형상마다 wide16_action primitive rollout 에서 32 step 유효 창 192 개를 뽑아 MJX 에서 orig / swap (다른 창의 A) / offset (±0.2 rad) 3 가지로 재생. 세 변형은 qpos·qvel 이 완전히 같다 (group_id 로 묶임).
- 기존 `action_pairs/` (e000~e002, 옛 rollouts/seq 소스, 2026-10-20) 와 섞지 않는다. 소스가 다르므로 v16 은 이 폴더 16 파일만 쓴다.
- 수집: `bash runs/pipeline_v1/collect_action_pairs_wide16.sh` (tmux `v16_collect_pairs`, 2026-11-02). kp30 (design-space 기본). 형상당 576 궤적, 약 3.3 MB.
