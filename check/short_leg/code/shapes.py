"""짧은 다리 단계 점검 (2026-11-12 사용자 결정): 허벅지·종아리를 같은 비율로 -5/-10/-15/-20/-25 %.
토큰 (check/slide/shapes.py 와 같은 환산): 허벅지 t3·t4 = 0.5 + 0.01×% (0.20 m 기준), 종아리 t9·t10 = 0.5 - 0.011667×% (0.175 m 기준).
L20 = check/slide 의 E1 과 같은 형상."""
BASE = [0.5] * 13
LEVELS = [5, 10, 15, 20, 25]


def short(p):
    e = list(BASE)
    e[3] = e[4] = round(0.5 + 0.01 * p, 4)
    e[9] = e[10] = round(0.5 - 0.035 / 3 * p, 4)
    return e


SHAPES = {f'L{p:02d}': dict(name=f'다리 짧은 형상 (허벅지·종아리 -{p} %)', E=short(p),
                           en=f'L{p:02d} short legs (-{p}%)',
                           dim=f'허벅지 {0.20 * (1 - p / 100):.3f} m · 종아리 {0.175 * (1 - p / 100):.3f} m (몸통 0.30 m)')
          for p in LEVELS}
MOTIONS = ['squat__r00', 'walkBackwardSlow__r00']
