"""계단을 얹은 CPU TrackEnv.

TT 의 ctrl 정책을 그대로 쓰되 지형만 계단으로 바꾼다. CPU MuJoCo 이므로 MJX 와 달리
cylinder-box 충돌 제약이 없어 로봇 전체가 계단과 정상적으로 충돌한다.

`post_process_mjcf` 는 `tokenize()` **이후**에 불리므로, 여기서 추가한 계단 geom 은
로봇 토큰에 섞이지 않는다.
"""

from dm_control import mjcf

from t2.env.mj_utils import default_root_element
from t2.env.track_env import FreeBaseTrackEnv
from t2.mjx_env.stair_env import DEFAULT_STAIRS, add_stairs


class StairFreeBaseTrackEnv(FreeBaseTrackEnv):
    def __init__(
        self,
        *args,
        num_stairs: int = DEFAULT_STAIRS["num_stairs"],
        stair_rise: float = DEFAULT_STAIRS["rise"],
        stair_run: float = DEFAULT_STAIRS["run"],
        stair_start_x: float = DEFAULT_STAIRS["start_x"],
        stair_width: float = DEFAULT_STAIRS["width"],
        stair_landing_length: float = DEFAULT_STAIRS["landing_length"],
        **kwargs,
    ):
        self.stairs = dict(
            num_stairs=num_stairs,
            rise=stair_rise,
            run=stair_run,
            start_x=stair_start_x,
            width=stair_width,
            landing_length=stair_landing_length,
        )
        super().__init__(*args, **kwargs)

    def post_process_mjcf(self, mjcf_model: mjcf.RootElement) -> mjcf.RootElement:
        mjcf_model = super().post_process_mjcf(mjcf_model)
        # 로봇 root 의 worldbody 에 직접 붙이면 관측이 계단 geom 11 개를 각각
        # free link 로 세어 shape 이 깨진다. `add_plane` 과 같은 방식으로 별도
        # root element 에 담아 attach 한다.
        stair_root = default_root_element("stairs")
        add_stairs(stair_root, **self.stairs)
        mjcf_model.attach(stair_root)
        return mjcf_model
