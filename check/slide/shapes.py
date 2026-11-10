"""점검용 형상 5 개 (E 13 차원, e000 = 0.5 기준 ±20 %) 와 대표 모션 5 개.
토큰: 2 몸통 길이 (0.35 → 0.24 m, 0.65 → 0.36 m), 3·4 앞·뒤 허벅지 (0.7 → 0.16 m 짧음, 0.3 → 0.24 m 김),
9·10 앞·뒤 종아리 (0.2667 → 0.14 m 짧음, 0.7333 → 0.21 m 김)."""
BASE = [0.5] * 13


def shape(**tok):
    e = list(BASE)
    for k, v in tok.items():
        e[int(k[1:])] = v
    return e


SHAPES = {
    'E0': dict(name='일반 형상', E=BASE),
    'E1': dict(name='다리 짧은 형상 (허벅지·종아리 -20 %)', E=shape(t3=0.7, t4=0.7, t9=0.2667, t10=0.2667)),
    'E2': dict(name='다리 긴 형상 (허벅지·종아리 +20 %)', E=shape(t3=0.3, t4=0.3, t9=0.7333, t10=0.7333)),
    'E3': dict(name='몸통 짧은 형상 (-20 %)', E=shape(t2=0.35)),
    'E4': dict(name='몸통 긴 형상 (+20 %)', E=shape(t2=0.65)),
}
MOTIONS = ['squat__r00', 'walkForwardSlow__r00', 'trotForwardFast__r00', 'trotCCWSlow__r00', 'walkBackwardSlow__r00']
