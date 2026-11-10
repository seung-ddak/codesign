"""Balanced source-motion assignments and a 13-dimensional morphology catalog."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allocate(motions, designs=1000, common=30, extra=70, seed=0):
    if len(motions)<common+extra or designs<1 or common<12 or extra<1:
        raise ValueError('insufficient motions or invalid allocation sizes')
    ids=[m['id'] for m in motions]
    if len(set(ids))!=len(ids):raise ValueError('duplicate motion IDs')
    rng=np.random.default_rng(seed);by_k={k:[] for k in range(1,13)}
    for m in motions:by_k[m['k']].append(m['id'])
    shared=[]
    # Equal K coverage, then spread the remaining common slots over K.
    quotas=np.full(12,common//12);quotas[np.linspace(0,11,common%12,dtype=int)]+=1
    for k,n in enumerate(quotas,1):
        if len(by_k[k])<n:raise ValueError('not enough source motions for every K')
        shared.extend(int(x) for x in rng.choice(by_k[k],int(n),replace=False))
    pool=np.array(sorted(set(ids)-set(shared)));cycles=int(np.ceil(designs*extra/len(pool)))
    stream=np.concatenate([rng.permutation(pool) for _ in range(cycles)])
    for start in range(0,designs*extra,extra):
        seen=set()
        for j in range(start,start+extra):
            if int(stream[j]) in seen:
                swap=next((t for t in range(j+1,len(stream)) if int(stream[t]) not in seen and stream[t] not in stream[j+1:start+extra]),None)
                if swap is None:raise ValueError('cannot balance allocation')
                stream[j],stream[swap]=stream[swap],stream[j]
            seen.add(int(stream[j]))
    rows=[dict(design=f'e{i:04d}',common=shared.copy(),extra=stream[i*extra:(i+1)*extra].tolist()) for i in range(designs)]
    counts=np.array([np.count_nonzero(stream[:designs*extra]==i) for i in pool])
    assert counts.max()-counts.min()<=1
    assert all(len(set(r['common']+r['extra']))==common+extra for r in rows)
    return shared,rows,dict(extra_reuse_min=int(counts.min()),extra_reuse_max=int(counts.max()),extra_pool_size=len(pool))


# Token order is the range-field consumption order of topology 00000011.
LENGTH_TOKENS={2:('body_length_m','base_geom_1',2.),3:('front_thigh_m','left_leg@calf',1.),
               4:('rear_thigh_m','left_leg_1@calf',1.),9:('front_calf_m','right_leg@calf_geom',1.),
               10:('rear_calf_m','right_leg_1@calf_geom',1.)}


def length_ranges():
    import xml.etree.ElementTree as ET
    path=Path(__file__).resolve().parents[3]/'transformer-transformer/assets/mjcf/umi_on_legs_plus_plus/quadruped_manipulator.xml'
    fields=[n for n in ET.parse(path).getroot().find('custom').findall('numeric') if '\\range\\' in n.get('name')]
    result={}
    for i,(name,identifier,scale) in LENGTH_TOKENS.items():
        field=fields[i]
        if field.get('name').split('\\')[0]!=identifier:raise ValueError('robot token order changed')
        values=np.fromstring(field.get('data'),sep=' ')
        if values.shape!=(2,) or values.mean()==0:raise ValueError('expected a nonzero scalar length range')
        result[i]=values
    return result


def physical_dimensions(E,ranges=None):
    ranges=length_ranges() if ranges is None else ranges
    return {LENGTH_TOKENS[i][0]:float(abs(values[0]+E[i]*(values[1]-values[0]))*LENGTH_TOKENS[i][2]) for i,values in ranges.items()}


def morphologies(count=1000,seed=0):
    active=list(LENGTH_TOKENS);levels=min(20,(count-1)//(2*len(active)))
    if levels<1:raise ValueError('need baseline and positive/negative variations of all five lengths')
    rng=np.random.default_rng(seed);ranges=length_ranges();rows=[]
    def add(kind,deltas):
        E=np.full(13,.5)
        for i,d in deltas.items():
            low,high=ranges[i];value=(low+high)*.5*(1+d);E[i]=(value-low)/(high-low)
        if np.any(E<0) or np.any(E>1):raise ValueError('requested physical range exceeds robot design space')
        rows.append(dict(design=f'e{len(rows):04d}',E=E.tolist(),topology='00000011',variation=kind,
                         changed_tokens=sorted(deltas),relative_dimension_changes={str(i):float(d) for i,d in deltas.items()},
                         physical_dimensions=physical_dimensions(E,ranges)))
    add('baseline',{})
    for i in active:
        for magnitude in np.linspace(.2/levels,.2,levels):
            for sign in [-1,1]:add('single_token',{i:float(sign*magnitude)})
    while len(rows)<count:
        k=2+(len(rows)-1-2*levels*len(active))%4
        selected=rng.choice(active,k,replace=False)
        deltas={int(i):float(rng.choice([-1,1])*rng.uniform(.01,.2)) for i in selected}
        add('multiple_tokens',deltas)
    assert len({tuple(r['E']) for r in rows})==count
    return rows


def build(source,out,count=1000,seed=0):
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('manifest already exists; use existing immutable assignment')
    status=json.loads((source/'status.json').read_text())
    if status.get('status')!='complete' or status['accepted']!=10000:raise ValueError('wait for all 10,000 accepted source motions')
    records=[json.loads(s) for s in (source/'records.jsonl').read_text().splitlines()];accepted={r['id']:r for r in records if r['accepted']}
    if len(accepted)!=10000:raise ValueError('journal/status mismatch')
    motions=[]
    for i,r in sorted(accepted.items()):
        p=source/r['file']
        if not p.is_file():raise FileNotFoundError(p)
        motions.append(dict(id=i,k=r['plan']['k'],seconds=r['plan']['seconds'],file=str(p),sha256=digest(p)))
    shared,assignments,stats=allocate(motions,count,seed=seed)
    catalog=morphologies(count,seed)
    for e,a in zip(catalog,assignments):e.update(common=a['common'],extra=a['extra'])
    result=dict(version=2,source=str(source),source_config_sha256=digest(source/'collection_config.json'),seed=seed,
                topology='00000011',physical_relative_range=[-.2,.2],active_length_tokens=list(LENGTH_TOKENS),
                sampling='e000 baseline + one-token signed physical length sweeps + simultaneous two-to-five-token length changes; zero/removed tokens fixed',
                baseline_design='e000',baseline_E=[.5]*13,baseline_dimensions=catalog[0]['physical_dimensions'],
                variation_counts=dict(collections.Counter(r['variation'] for r in catalog)),
                fixed_tokens=dict(removed_arm=[0,11,12],zero_battery_position=[1],zero_leg_yaw=[5,6,7,8]),
                common_motion_ids=shared,motions=motions,designs=catalog,assignment_stats=stats,
                total_requested_rollouts=count*100,split_note='Use source motion group_id across morphologies for TT splits; common references are not identical S0/action interventions.')
    tmp=out/'manifest.json.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(out/'manifest.json')
    (out/'designs.json').write_text(json.dumps({e['design']:e['E'] for e in catalog},indent=2))
    return result


def prepare_reference(source,destination):
    from scipy.spatial.transform import Rotation,Slerp
    from collect_pd_rollouts import save_rollout
    source=Path(source);destination=Path(destination)
    with np.load(source) as z:
        root=z['root'];ts=z['time'];q=z['q'];names=z['joint_names'].tolist()
        if root.shape!=(len(ts),13) or q.shape!=(len(ts),12) or not np.all(np.isfinite(root)) or not np.all(np.isfinite(q)):
            raise ValueError('invalid source trace')
        td=np.arange(round(float(ts[-1])*30)+1)/30
        if not np.all(np.diff(ts)>0) or not np.isclose(ts[-1],10 if len(ts)==301 else 20):raise ValueError('invalid time axis')
        order=[names.index(leg+'_'+joint+'_joint') for leg in ['FR','FL','RR','RL'] for joint in ['hip','thigh','calf']]
        angles=np.stack([np.interp(td,ts,q[:,j]) for j in order],axis=1)
        pos=np.stack([np.interp(td,ts,root[:,j]) for j in range(3)],axis=1)
        quat=Slerp(ts,Rotation.from_quat(root[:,3:7]))(td).as_quat()[:,[3,0,1,2]]
        near=np.minimum(np.round(td*30).astype(int),len(ts)-1)
        foot=np.stack([np.stack([np.interp(td,ts,z['foot_pos'][:,f,j]) for j in range(3)],axis=1) for f in range(4)],axis=1)
        save_rollout(destination,dict(fix_q=angles,fix_base_pos=pos,fix_base_quat=quat,fix_foot_pos=foot-[0,0,.02],
                                    fix_contact=z['foot_floor_contact'][near],dt=1/30,source_motion=str(source),source_sha256=digest(source)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--designs',type=int,default=1000);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    result=build(a.source,a.out,a.designs,a.seed);print(result['assignment_stats'])

if __name__=='__main__':main()
