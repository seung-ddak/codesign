"""형상 토큰 환산 (check/slide/shapes.py 와 같은 기준).
다리 길이 변화 d % (허벅지·종아리 같은 비율, 0.20 m · 0.175 m 기준): 허벅지 t3·t4 = 0.5 - 0.01·d, 종아리 t9·t10 = 0.5 + 0.035/3·d.
몸통 길이 변화 b % (0.30 m 기준): t2 = 0.5 + 0.0075·b."""
BASE = [0.5] * 13


def token(leg=0., body=0.):
    e = list(BASE)
    e[3] = e[4] = round(0.5 - 0.01 * leg, 5)
    e[9] = e[10] = round(0.5 + 0.035 / 3 * leg, 5)
    e[2] = round(0.5 + 0.0075 * body, 5)
    assert all(0 <= v <= 1 for v in e), (leg, body)
    return e


def name(leg, body):
    return f'G_l{leg:+03.0f}_b{body:+03.0f}'


def dims(leg, body):
    return f'허벅지 {0.20 * (1 + leg / 100):.3f} m · 종아리 {0.175 * (1 + leg / 100):.3f} m · 몸통 {0.30 * (1 + body / 100):.3f} m'


