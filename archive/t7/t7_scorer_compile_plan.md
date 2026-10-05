## T7 DesignScorer JAX/XLA 컴파일 최적화 계획

### 1. 현재 문제

현재 T7에서 가장 큰 문제는 DDPO trajectory 저장이 아니라 `DesignScorer`의 첫 JAX/XLA 컴파일이다.

관찰된 현상은 다음과 같다.

- Zarr trajectory buffer는 약 389 MiB에서 정상 생성 완료
- 이후 trajectory buffer 쓰기는 더 이상 증가하지 않음
- Python RSS는 약 22 GiB까지 계속 증가
- CPU 사용률은 약 180%
- GPU 메모리는 약 892 MiB 수준
- `history.jsonl`은 0줄
- 결국 프로세스가 종료됨

따라서 DDPO generation 및 trajectory 저장은 완료되었지만, 첫 `scorer.score()`에서 JAX/XLA가 지나치게 큰 rollout graph를 컴파일하는 과정에서 host RAM이 증가한 것으로 판단한다.

---

## 2. 현재 DesignScorer 구조의 문제

현재 `DesignScorer`는 모든 structure group을 하나의 `MultiDesignEnv`에 넣고 하나의 `jax.jit()` 함수로 전체 rollout을 컴파일한다.

현재 구조는 다음과 같다.

```text
전체 structure groups
        ↓
각 group마다 slots개 morphology
        ↓
각 morphology마다 7개 EVAL_COMMANDS
        ↓
전체를 하나의 MultiDesignEnv로 결합
        ↓
하나의 jax.jit(_rollout_impl)
```

현재 design space에는 32개의 structure group이 존재한다.

기본 설정이

```text
structure groups = 32
slots = 8
commands = 7
```

이므로 하나의 JIT graph 내부에 포함되는 rollout slot 수는

\[
32\times8\times7=1792
\]

개이다.

즉 DDPO가 실제로 한 update에서 morphology 8개만 생성했더라도, 현재 scorer는 사용하지 않는 structure group까지 포함한 거대한 1792-env graph를 첫 호출 시 모두 컴파일한다.

이 때문에 GPU 메모리가 아니라 XLA compilation 과정의 host RAM 사용량이 크게 증가한다.

---

## 3. 최적화 핵심 방향

최종 버전에서는 평가 조건이나 reward 정의를 줄이지 않는다.

기존과 동일하게

\[
R(E)
=
\frac{1}{7}
\sum_{c\in C}
J(E,\theta_{\mathrm{fixed}},c)
\]

를 사용한다.

변경하는 것은 **컴파일 단위**이다.

기존:

```text
32 groups
×
8 slots
×
7 commands
↓
하나의 giant JAX graph
```

수정:

```text
현재 필요한 structure group 하나
×
8 slots
×
command 하나
↓
작은 JAX graph
```

로 변경한다.

따라서 evaluation 결과는 동일하게 유지하면서 XLA가 한 번에 처리해야 하는 graph 크기만 크게 줄인다.

---

## 4. MultiDesignEnv 기반 전체 동시 평가 제거

T7의 `DesignScorer`에서는 더 이상 모든 structure group을 하나의 `MultiDesignEnv`로 합치지 않는다.

`MultiDesignEnv`는 T5처럼 여러 morphology를 하나의 PPO training batch에서 동시에 학습하기 위한 구조에는 적합하지만, T7에서는 소수의 새 morphology를 반복적으로 평가하는 것이 목적이다.

따라서 scorer에서는 structure group 단위 평가 구조로 변경한다.

최종적으로 다음과 같은 구조를 사용한다.

```text
DesignScorer
│
├─ Frozen locomotion policy πθ
│
├─ structure template 정보
│
└─ GroupScorer cache
    ├─ GroupScorer(group A)
    ├─ GroupScorer(group B)
    ├─ GroupScorer(group C)
    └─ ...
```

각 `GroupScorer`는 하나의 structure group에 대해서만 MJX environment를 구성하고 JIT compile한다.

---

## 5. Lazy Group Compilation

32개의 structure group을 프로그램 시작 시 모두 compile하지 않는다.

DDPO가 생성한 morphology를 먼저 `structure_key` 기준으로 분류한다.

예:

```text
E1 → group 3
E2 → group 3
E3 → group 11
E4 → group 18
E5 → group 11
E6 → group 3
E7 → group 25
E8 → group 18
```

이 batch에서 실제 필요한 group은

```text
group 3
group 11
group 18
group 25
```

뿐이다.

따라서 처음 등장한 group에 대해서만 `GroupScorer`를 생성하고 JIT compile한다.

전체 흐름은 다음과 같다.

```text
Generated morphologies
        ↓
structure_key로 분류
        ↓
해당 group scorer가 이미 존재하는가?
        ↓
Yes → 기존 compiled executable 재사용
No  → 해당 group만 새로 compile
```

이를 통해 사용되지 않는 structure group을 컴파일하는 낭비를 제거한다.

---

## 6. 7개 command도 동시에 컴파일하지 않음

현재 한 group 안에서도

\[
slots\times7
\]

개의 environment를 동시에 구성한다.

예를 들어 `slots=8`이면

\[
8\times7=56
\]

개의 environment가 한 group의 graph에 포함된다.

최종 버전에서는 command도 순차적으로 평가한다.

```text
Group A의 morphology 8개

command 1
→ 8개 morphology rollout

command 2
→ 동일한 8개 morphology rollout

...

command 7
→ 동일한 8개 morphology rollout
```

마지막에 각 morphology에 대해 7개의 return 평균을 계산한다.

\[
R_i
=
\frac1{7}
\sum_{c=1}^{7}R_{i,c}
\]

따라서 reward의 의미는 기존 방식과 완전히 동일하다.

---

## 7. Command는 JIT의 Dynamic Input으로 사용

command별로 별도의 JIT function을 생성하면 안 된다.

즉 다음과 같은 방식은 피한다.

```text
command 1용 JIT
command 2용 JIT
...
command 7용 JIT
```

대신 하나의 rollout function을 만들고 command를 입력 tensor로 전달한다.

개념적으로:

```python
@jax.jit
def rollout(designs, command, rng):
    ...
```

형태로 만든다.

각 command 호출에서는 값만 변경한다.

```text
rollout(designs, command_1, rng)
rollout(designs, command_2, rng)
...
rollout(designs, command_7, rng)
```

command tensor의 shape와 dtype은 항상 동일하게 유지한다.

예:

```text
shape = (slots, 4)
dtype = float32
```

따라서 최초 한 번 compile된 executable을 7개 command 모두 재사용한다.

---

## 8. Slot 수는 고정 Shape으로 유지

한 structure group에 실제 morphology가 3개밖에 없더라도 JIT input shape을 3으로 만들지 않는다.

항상 고정된

```text
slots = 8
```

형태를 유지한다.

예:

```text
실제 morphology
E1
E2
E3

나머지 slot
template
template
template
template
template
```

평가 후 실제 morphology에 해당하는 첫 3개 결과만 사용하고 나머지 template result는 무시한다.

이를 통해 batch마다 morphology 개수가 달라져도 input shape가 변하지 않는다.

즉

```text
이번 batch → 3개 morphology
다음 batch → 6개 morphology
다음 batch → 8개 morphology
```

여도 JAX가 보는 shape는 항상 동일하다.

따라서 morphology 수 변화에 따른 재컴파일을 방지한다.

---

## 9. JAX graph 크기 변화

현재 구조의 최초 compile 규모:

\[
32\times7\times8
=
1792
\]

rollout slots.

수정 후 하나의 compile 규모:

\[
1\times1\times8
=
8
\]

rollout slots.

즉 XLA가 한 번에 tracing 및 optimization 해야 하는 graph가 매우 작아진다.

정확히 1792/8 배만큼 compile memory가 감소한다고 단순 계산할 수는 없지만, graph 복잡도를 구조적으로 크게 줄일 수 있다.

---

## 10. lax.scan은 유지

현재 locomotion rollout의 긴 time horizon은 `jax.lax.scan()`으로 처리한다.

이 구조는 유지한다.

즉 다음 부분은 문제로 보지 않는다.

```text
1000 simulation steps
↓
lax.scan
```

문제는 scan 자체가 아니라 scan body 내부에

```text
32 groups
×
8 slots
×
7 commands
```

를 동시에 넣은 것이다.

따라서

```text
large scan body
```

를

```text
small scan body
```

로 만드는 것이 핵심이다.

---

## 11. GroupScorer Cache

한 번 compile한 structure group scorer는 메모리 또는 persistent cache에 보존한다.

예:

```text
group_cache = {
    structure_key_A: scorer_A,
    structure_key_B: scorer_B,
    structure_key_C: scorer_C
}
```

새로운 DDPO batch에서 동일한 structure group이 다시 등장하면 새로 compile하지 않고 기존 executable을 재사용한다.

따라서 학습이 진행될수록 compile overhead는 감소한다.

예:

```text
Update 1
group A 최초 등장 → compile
group B 최초 등장 → compile

Update 2
group A 재등장 → reuse
group C 최초 등장 → compile

Update 3
group A → reuse
group B → reuse
group C → reuse
```

형태가 된다.

---

## 12. Persistent JAX Compilation Cache 사용

프로세스를 종료했다가 다시 실행해도 이미 컴파일한 executable을 재사용할 수 있도록 JAX persistent compilation cache를 사용한다.

예:

```bash
mkdir -p ~/quad_ws/jax_cache
export JAX_COMPILATION_CACHE_DIR="$HOME/quad_ws/jax_cache"
```

또는 Python startup 단계에서 compilation cache directory를 지정한다.

이 기능은 첫 compile 자체를 작게 만드는 해결책은 아니다.

따라서 순서는 반드시

```text
1. compile graph 자체를 작게 만든다.
2. 작은 compiled executable을 persistent cache에 저장한다.
```

로 진행한다.

---

## 13. 최종 DesignScorer 처리 흐름

DDPO가 한 batch의 morphology를 생성하면 다음과 같이 처리한다.

```text
Generated morphologies
        ↓
structure_key 기준 grouping
        ↓

Group A
E1, E3, E5

Group B
E2, E7

Group C
E4, E6, E8
```

각 group에 대해:

```text
Group A

slots=8 고정
↓
E1, E3, E5 + template fill
↓
command 1 rollout
↓
command 2 rollout
↓
...
↓
command 7 rollout
↓
각 morphology의 7-command 평균 return
```

그다음 Group B, Group C도 동일하게 수행한다.

마지막에 모든 morphology의 return을 원래 DDPO batch 순서로 합친다.

```text
R1
R2
R3
...
RB
```

이 값을 DDPO advantage 계산에 사용한다.

---

## 14. 기존 reward 및 평가 조건은 유지

이 최적화는 evaluation fidelity를 낮추기 위한 방법이 아니다.

다음 조건은 그대로 유지한다.

- frozen \(\pi_\theta\)
- 동일 MJX physics
- 동일 episode length
- 동일 7개 EVAL_COMMANDS
- 동일 morphology
- 동일 task reward
- 동일 7-command mean return

즉 변경되는 것은

```text
평가 내용
```

이 아니라

```text
JAX/XLA가 평가를 컴파일하는 단위
```

뿐이다.

---

## 15. 메모리 문제가 해결되는 이유

기존에는 XLA가 한 번에

```text
32 structure groups
×
8 design slots
×
7 commands
×
full MJX dynamics
×
locomotion policy
×
1000-step scan
```

을 하나의 computational graph로 분석하고 최적화해야 했다.

이 과정에서 intermediate representation 및 compilation data가 CPU RAM에 대량으로 생성되었다.

수정 후에는 XLA가 한 번에 다루는 것은

```text
1 structure group
×
8 design slots
×
1 command
×
full MJX dynamics
×
locomotion policy
×
1000-step scan
```

뿐이다.

따라서 compile-time host memory 사용량이 크게 감소한다.

직관적으로는

```text
1792개의 문제를 한 번에 컴파일
```

하던 것을

```text
8개의 문제씩 나누어 컴파일하고
같은 compiled program을 반복 사용
```

하는 것이다.

---

## 16. 최종 목표 구조

최종 scorer는 다음 조건을 만족해야 한다.

- 모든 structure group을 동시에 compile하지 않음
- 현재 DDPO batch에 등장한 group만 평가
- 처음 등장한 group만 lazy compile
- 동일 group은 compiled executable 재사용
- 7 commands는 동일 JIT function에 dynamic input으로 전달
- slot 수는 항상 고정
- 7-command 평균 reward는 기존과 동일
- `lax.scan` rollout 유지
- persistent JAX compilation cache 사용
- first XLA compile 중 CPU RAM OOM 방지

최종 구조를 한 줄로 표현하면 다음과 같다.

\[
\boxed{
\text{Global Multi-Group JIT}
\rightarrow
\text{Lazy Per-Group Fixed-Shape JIT}
}
\]

즉 **평가량을 줄이는 것이 아니라, 동일한 평가를 작은 고정 크기의 compiled graph로 분할하여 수행하는 것**이 핵심 해결 방향이다.