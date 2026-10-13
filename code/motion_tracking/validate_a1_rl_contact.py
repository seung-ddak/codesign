"""Compare recorded A1 RL force labels with FK foot clearance; no dynamics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from pd_replay import build_model, LEGS, RADIUS, SIM_MOTIONS


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / 'runs/TT_retarget/a1_rl_contact'
    out.mkdir(parents=True, exist_ok=True)
    raw_path, (start, end) = SIM_MOTIONS['jumpInPlace']
    raw = np.load(root / raw_path)
    ref = np.load(root / 'artifacts/pd_replay/jumpInPlace.npz')
    model = build_model()
    data = mujoco.MjData(model)
    feet = [model.geom(f'{leg}_foot').id for leg in LEGS]
    raw_qpos = np.c_[raw['base_pos'][start:end], raw['base_quat'][start:end], raw['q'][start:end]]
    raw_foot = []
    for qpos in raw_qpos:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        raw_foot.append(data.geom_xpos[feet, 2].copy()-RADIUS)
    raw_foot = np.array(raw_foot)
    rc = raw['contact'][start:end].astype(bool)
    c = ref['fix_contact'].astype(bool)
    rt = np.arange(len(rc))*float(raw['dt'])
    t = np.arange(len(c))*float(ref['dt'])
    nearest = np.minimum(np.round(t/float(raw['dt'])).astype(int), len(rc)-1)
    np.testing.assert_array_equal(c, rc[nearest])
    origin = t[np.flatnonzero(~c.any(axis=1))[0]]
    rl = LEGS.index('RL')
    foot = ref['fix_foot_pos'][:,:,2]
    for k in range(len(t)):
        data.qpos[:] = np.r_[ref['fix_base_pos'][k], ref['fix_base_quat'][k], ref['fix_q'][k]]
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(data.geom_xpos[feet,2]-RADIUS, foot[k], atol=1e-7)
    report = {'origin_clip_s': float(origin), 'raw_frame_window': [start,end],
              'label_resampling_exact_match': True, 'datasets': {}}
    fig, ax = plt.subplots(3, 1, figsize=(12,10), sharex=True)
    for name, times, z, labels, base, color in [
        ('raw_30ms_FK', rt, raw_foot, rc, raw_qpos[:,2], 'gray'),
        ('reference_20ms', t, foot, c, ref['fix_base_pos'][:,2], 'darkorange')]:
        x = times-origin
        peak = int(np.argmax(base))
        window = (x >= -.2) & (x <= .3)
        contact = labels[:,rl]
        stats = {'body_peak_time_relative_s': float(x[peak]), 'body_peak_z_mm': float(base[peak]*1000),
                 'RL_at_body_peak_height_mm': float(z[peak,rl]*1000),
                 'RL_at_body_peak_contact': bool(contact[peak]),
                 'all_feet_at_body_peak_height_mm': (z[peak]*1000).tolist(),
                 'all_feet_at_body_peak_contact': labels[peak].tolist(),
                 'contact_true_RL_max_clearance_mm': float(z[contact,rl].max()*1000),
                 'threshold_counts': {str(mm): {'contact_true_above_threshold_all': int((contact & (z[:,rl]>mm/1000)).sum()),
                     'contact_true_above_threshold_jump_window': int((window & contact & (z[:,rl]>mm/1000)).sum())}
                     for mm in [1,5,10,20]}}
        report['datasets'][name] = stats
        ax[0].plot(x, base*1000, '.-', color=color, label=name)
        ax[1].plot(x, z[:,rl]*1000, '.-', color=color, label=f'{name} RL clearance')
        ax[1].scatter(x[contact], z[contact,rl]*1000, s=45, facecolors='none', edgecolors=color, label=f'{name} contact=True')
        ax[2].step(x, contact.astype(int), where='mid', color=color, label=f'{name} RL label')
        np.savetxt(out / f'{name}.csv', np.c_[times,x,base,z,labels.astype(int)], delimiter=',',
                   header='clip_time_s,time_from_event_s,body_z_m,FR_z_m,FL_z_m,RR_z_m,RL_z_m,FR_contact,FL_contact,RR_contact,RL_contact', comments='')
    ax[0].set_ylabel('Body absolute height [mm]')
    ax[1].set_ylabel('RL foot bottom above ground [mm]')
    ax[1].axhline(0,color='black',lw=.8)
    ax[1].axhline(10,color='red',ls=':',label='10 mm diagnostic threshold')
    ax[2].set(ylabel='RL contact label', xlabel='Time from A1 first no-contact reference frame [s]', yticks=[0,1])
    for a in ax:
        a.axvline(0,color='gray',ls=':')
        a.grid(alpha=.2)
        a.legend(fontsize=8)
    ax[-1].set_xlim(-.25,.4)
    fig.suptitle('A1 RL contact label vs foot clearance\nFoot height: MuJoCo A1 FK; label: recorded Isaac Gym force > 1 N')
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(out / 'rl_contact_comparison.png', dpi=160)
    plt.close(fig)
    x = t-origin
    color = 'darkorange'
    def draw_leg(axes, index):
        leg = LEGS[index]
        contact = c[:,index]
        axes[0].plot(x, ref['fix_base_pos'][:,2]*1000, '.-', color=color)
        axes[0].set(title=f'A1 {leg} — 50 Hz reference only', ylabel='Body height [mm]')
        axes[1].plot(x, foot[:,index]*1000, '.-', color=color)
        axes[1].scatter(x[contact], foot[contact,index]*1000, s=45,
                        facecolors='none', edgecolors=color, label='Contact=True')
        axes[1].axhline(0, color='gray', lw=.7)
        axes[1].set_ylabel(f'{leg} foot clearance [mm]')
        axes[1].legend(fontsize=8)
        axes[2].step(x, contact.astype(int), where='mid', color=color)
        axes[2].set(ylabel=f'{leg} contact label', yticks=[0,1],
                    xlabel='Time from A1 first no-contact frame [s]')
        for axis in axes:
            axis.axvline(0, color='gray', ls=':')
            axis.grid(alpha=.2)
            axis.set_xlim(-.25,.4)
    for index, leg in enumerate(LEGS):
        fig, axes = plt.subplots(3, 1, figsize=(11,9), sharex=True)
        draw_leg(axes, index)
        fig.tight_layout()
        fig.savefig(out / f'{leg.lower()}_contact_reference_only.png', dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(22,10), sharex=True, sharey='row')
    for index in range(len(LEGS)):
        draw_leg(axes[:,index], index)
    fig.tight_layout()
    fig.savefig(out / 'all_feet_contact_reference_only.png', dpi=160)
    plt.close(fig)
    (out / 'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
