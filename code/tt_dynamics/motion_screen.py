"""Post-hoc movement and task checks; never change training or task reward."""
import json
from pathlib import Path
import numpy as np


def zigzag_y(x, amp, period):
    return amp * (2/np.pi) * np.arcsin(np.sin(2*np.pi*x/period))


def crosses_zigzag(points, amp, period, width):
    """Require both opposite peaks in order, then one period of net progress."""
    last_time = -1.
    for target in (period/4, 3*period/4, period):
        found = False
        for i in range(len(points)-1):
            x0, x1 = points[i, 0], points[i+1, 0]
            if x0 <= target <= x1 and x1 > x0:
                alpha = (target-x0)/(x1-x0)
                when = i + alpha
                y = points[i, 1] + alpha*(points[i+1, 1]-points[i, 1])
                if when > last_time and abs(y-zigzag_y(target, amp, period)) <= width+1e-6:
                    last_time = when; found = True; break
        if not found:
            return False
    return bool(points[-1, 0]-points[0, 0] >= period-1e-6)


def trajectory_metrics(states, mask, row, task, settings, seconds):
    initial = row.get('initial_physics', {}).get('physically_valid') is True
    length = int(mask.sum())
    valid = bool(initial and not row.get('execution_skipped', False) and length >= 2
                 and mask[:length].all() and not mask[length:].any() and np.isfinite(states[:length]).all()
                 and row.get('mjx_total') is not None and np.isfinite(row['mjx_total']))
    out = dict(valid=valid, survived=False, moving=False, task_success=False,
               progress_m=None, inside_lane_fraction=None, path_err_m=None, reward=None)
    if not valid:
        return out
    points = states[:length, :2]
    survived = bool(row.get('mjx_fell_step') == -1 and length == round(seconds/.02))
    width = settings['half_width']
    if task == 'zigzag':
        amp, period = settings['amp'], settings['period']
        deviation = abs(points[:, 1]-zigzag_y(points[:, 0], amp, period))
        progress = float(points[-1, 0]-points[0, 0])
        completed = crosses_zigzag(points, amp, period, width)
    elif task == 'circle':
        radius = settings['radius']
        relative = points-np.array([0., radius])
        a, b = relative[:-1], relative[1:]
        angles = np.arctan2(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0], (a*b).sum(1))
        progress = float(angles.sum()*radius)
        deviation = abs(np.linalg.norm(relative, axis=1)-radius)
        completed = bool(angles.sum() >= 2*np.pi-1e-6)
    else:
        raise ValueError(f'unsupported task {task}')
    inside = deviation <= width+1e-6
    out.update(survived=survived, moving=bool(progress >= .075*seconds),
               task_success=bool(survived and inside.all() and completed),
               progress_m=progress, inside_lane_fraction=float(inside.mean()),
               path_err_m=float(row['path_err_m']) if row.get('path_err_m') is not None and np.isfinite(row['path_err_m']) else None,
               reward=float(row['mjx_total']))
    return out


def screen(directory):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text())
    result = dict(task=summary['task'], seconds=summary['seconds'], scales={}, paired_guidance={})
    all_rows = {}
    for scale, rows in summary['runs'].items():
        with np.load(directory/f'{scale}.npz') as data:
            assert len(rows) == len(data['mjx']) == len(data['mjx_mask'])
            metrics = [trajectory_metrics(data['mjx'][i], data['mjx_mask'][i], row, summary['task'],
                                          summary['reward'], summary['seconds']) for i, row in enumerate(rows)]
        all_rows[scale] = metrics
        valid = [m for m in metrics if m['valid']]
        errors = [m['path_err_m'] for m in valid if m['path_err_m'] is not None and np.isfinite(m['path_err_m'])]
        count = len(rows)
        values = {key: sum(m[key] for m in metrics)/count if count else 0.
                  for key in ('valid', 'survived', 'moving', 'task_success')}
        error = float(np.median(errors)) if errors else None
        progress = float(np.median([m['progress_m'] for m in valid])) if valid else None
        result['scales'][scale] = dict(candidates=count, rates=values, median_path_error_m=error,
                                      median_progress_m=progress, candidates_metrics=metrics,
                                      provisional_screen_pass=bool(values['valid'] >= .75 and values['survived'] >= .75
                                          and error is not None and error < .15 and progress is not None
                                          and progress >= .075*summary['seconds']))
    if 'scale_0' in all_rows:
        baseline = all_rows['scale_0']
        for scale, rows in all_rows.items():
            if scale == 'scale_0': continue
            if len(rows) != len(baseline): raise ValueError('paired candidate counts differ')
            delta = [r['reward']-b['reward'] for r,b in zip(rows, baseline)
                     if r['valid'] and b['valid'] and r['reward'] is not None and b['reward'] is not None
                     and np.isfinite(r['reward']) and np.isfinite(b['reward'])]
            result['paired_guidance'][scale] = dict(paired_valid=len(delta),
                median_reward_delta=float(np.median(delta)) if delta else None)
    return result
