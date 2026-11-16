"""여러 방향이 섞인 (3~5 방향) 새 CALM 중 s7 에서 학습 짝 형상으로 통과한 모션, 묶음마다 1 개 (방향 수 많은 순, 같으면 관절 오차 작은 순).
three_calm 과 같은 영상 3 개씩 (A1 단독 · A1+학습 형상 TT · A1+학습+미사용 형상 G_l+20_b+10). s7 과 같게 앞 10 초.
출력: runs/s7_eval/videos/good_mixed/<모션>/"""
import json
from common import RUNS
import three_calm
S5 = RUNS / 's5_retarget'; SH = json.load(open(S5 / 'assignment.json'))['shapes']
PICK = [('calm_round_00_00884', 'trotBackwardSlow__r00', 'G_l-20_b+00', 'Backward + CCW + Right'),
        ('calm_round_01_01509', 'trotCCWSlow__r00', 'G_l+20_b+20', 'CCW + CW + Right + Forward + Backward'),
        ('calm_round_01_01551', 'trotCWSlow__r00', 'G_l-20_b+20', 'CW + Left + Forward + Right'),
        ('calm_round_00_00178', 'trotForwardSlow__r00', 'G_l+00_b+00', 'Forward + Right + Backward'),
        ('calm_round_01_00227', 'trotLeftSlow__r00', 'G_l+00_b+20', 'Left + Right + Forward + CW'),
        ('calm_round_02_00650', 'trotRightSlow__r00', 'G_l+20_b-20', 'Right + CW + Backward + CCW + Left')]
if __name__ == '__main__':
    for M, G, s, dirs in PICK:
        three_calm.main(M, G, (s, SH[s]['E']), RUNS / 's7_eval/videos/good_mixed' / M, dirs, 10.0)
