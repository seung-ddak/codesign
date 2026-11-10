# 관리자 권한 없이 받은 OSMesa (quad_ws/.local_osmesa) 로 MuJoCo 메쉬 렌더링
L=/home/cgr-ugrad-2026/quad_ws/.local_osmesa/root/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$L${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYOPENGL_OSMESA_LIBRARY=$L/libOSMesa.so.8
