"""Run the supplied CALM A1 viewer with the matching checkpoint configs."""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('controller',choices=['llc','hlc','joystick'])
    p.add_argument('--calm-root',type=Path,default=Path(__file__).resolve().parents[4]/'external/a1-calm-main')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--sync-reset',action='store_true')
    p.add_argument('--disable-self-collision',action='store_true',help='Explicit physics diagnostic; leaves original files unchanged')
    a=p.parse_args();root=a.calm_root.resolve()
    os.environ['CALM_ROOT']=str(root);os.chdir(root);sys.path.insert(0,str(root/'calm'))
    import run  # Isaac Gym must load before torch.
    from reset_observations import install
    if a.disable_self_collision:
        from env.tasks.humanoid import Humanoid
        Humanoid._get_humanoid_collision_filter=lambda self: 1
    if a.sync_reset:
        install(root/'assets/urdf/a1_description/urdf/a1.urdf')
    llc=a.controller=='llc'
    script='fixed_latent_per_env_test.py' if llc else 'play_hlc_joystick.py' if a.controller=='joystick' else 'run.py'
    sys.argv=[script,'--task','QuadrupedAMPGetup' if llc else 'QuadrupedHeading','--test',
              '--cfg_env','calm/data/cfg/'+('a1_getup_31psc8n_4096.yaml' if llc else 'a1_heading_lat_up_c8_hit2.yaml'),
              '--cfg_train','calm/data/cfg/train/rlg/'+('calm_a1_31psc8n.yaml' if llc else 'hrl_a1_heading_lat31zs.yaml'),
              '--motion_file','calm/data/motions_a1/v2v3_31motions_squat2s_bfcut.pt',
              '--num_envs','31' if llc else '1',
              '--checkpoint','checkpoints/'+('lowLevelController.pth' if llc else 'highLevelPolicy.pth')]
    if not llc:
        sys.argv+=['--llc_checkpoint','checkpoints/lowLevelController.pth']
    if a.headless:sys.argv+=['--headless']
    runpy.run_path(str(root/'calm'/script),run_name='__main__')


if __name__=='__main__':main()
