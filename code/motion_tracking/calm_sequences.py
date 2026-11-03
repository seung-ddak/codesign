"""Duration-aware CALM command plans, validation, deduplication, and static coverage plots."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NAMES = ['forward','backward','left','right','cw_turn','ccw_turn',
         'left_then_forward','right_then_forward','left_then_backward','right_then_backward','sit','stand_up']
FPS = 30

def frame_count(k):
    return FPS * (20 if k >= 6 else 10)
# Includes the transition; compound actions need both a turn and translation phase.
MIN_FRAMES = np.array([12]*4+[15]*2+[24]*4+[48,45])


def valid_order(ids):
    if not 1 <= len(ids) <= 12 or len(set(ids)) != len(ids) or any(i not in range(12) for i in ids):
        return False
    seated = ids[0] == 11
    for i in ids:
        if i == 10:
            if seated: return False
            seated = True
        elif i == 11:
            if not seated: return False
            seated = False
        elif seated:
            return False
    return int(MIN_FRAMES[ids].sum()) <= frame_count(len(ids))


def make_sequence(seed, k, usage=None):
    if seed < 0 or k not in range(1,13):
        raise ValueError('nonnegative seed and K in 1..12 required')
    rng = np.random.default_rng(seed)
    choices = []
    for _ in range(600):
        ids = rng.choice(12, k, replace=False).tolist()
        if not valid_order(ids): continue
        lengths = MIN_FRAMES[ids].copy()
        remaining=frame_count(k)-int(lengths.sum())
        moving=np.array(ids)<10
        if moving.any():
            # Do not stretch a one-shot sit/stand into a long stationary hold.
            extra=np.zeros(k,dtype=int)
            for j in np.flatnonzero(~moving):
                extra[j]=int(rng.integers(0,min(6,remaining)+1));remaining-=extra[j]
            extra[moving]=rng.multinomial(remaining,rng.dirichlet(np.ones(int(moving.sum()))*2))
            lengths+=extra
        else:
            lengths+=rng.multinomial(remaining,rng.dirichlet(np.ones(k)*2))
        cost = rng.random()
        if usage is not None:
            next_usage = np.asarray(usage,dtype=float).copy()
            next_usage[ids] += lengths
            cost = float(np.var(next_usage))
        choices.append((cost, ids, lengths))
        if len(choices) == 8: break
    if not choices:
        raise RuntimeError('cannot construct a physically ordered sequence')
    _,ids,lengths = min(choices,key=lambda v:v[0])
    segments=[]; start=0
    for i,length in zip(ids,lengths):
        segments.append(dict(index=i,start=start,end=start+int(length),
                             speed=float(rng.uniform(.15,.55)) if i<10 else 0.,
                             turn_angle=float(rng.uniform(.15,.65)) if 4<=i<10 else 0.,
                             ramp_seconds=float(rng.uniform(.12,.3))))
        start+=int(length)
    signature=hashlib.sha256(json.dumps(segments,sort_keys=True).encode()).hexdigest()
    return dict(seed=int(seed),k=k,frames=frame_count(k),seconds=frame_count(k)/FPS,initial_pose='seated' if ids[0]==11 else 'standing',
                segments=segments,signature=signature)


def make_plan(count=10000,seed=0):
    if count<1 or seed<0: raise ValueError('positive count and nonnegative seed required')
    rng=np.random.default_rng(seed)
    ks=np.tile(np.arange(1,13),int(np.ceil(count/12)))[:count];rng.shuffle(ks)
    usage=np.zeros(12);seen=set();plans=[]
    for slot,k in enumerate(ks):
        attempt=0
        while True:
            p=make_sequence(seed+slot*1009+attempt,int(k),usage)
            if p['signature'] not in seen:break
            attempt+=1
        p['id']=slot;seen.add(p['signature']);plans.append(p)
        for segment in p['segments']:usage[segment['index']]+=segment['end']-segment['start']
    return plans


def labels(plan):
    frames=plan['segments'][-1]['end']
    result=np.full(frames,-1,dtype=np.int16)
    for s in plan['segments']:result[s['start']:s['end']]=s['index']
    if (result<0).any():raise ValueError('plan does not cover its complete duration')
    return result


def command(plan,step,entry_yaw,previous_command):
    """Return world vx/vy, facing yaw; pose actions are handled by LLC latents."""
    segment=next(s for s in plan['segments'] if s['start']<=step<s['end'])
    i=segment['index'];elapsed=(step-segment['start'])/FPS
    duration=(segment['end']-segment['start'])/FPS
    facing=entry_yaw; speed=segment['speed'];angle=0.
    if i>=10:return np.array([0.,0.,facing])
    if i in (4,5):
        facing += (-1 if i==4 else 1)*segment['turn_angle']*min(elapsed/max(duration*.8,1/FPS),1.)
        speed=0.
    elif i in (6,7,8,9):
        turn_duration=duration*.4
        facing += (1 if i in (6,8) else -1)*segment['turn_angle']*min(elapsed/turn_duration,1.)
        if elapsed<turn_duration:speed=0.
        if i in (8,9):angle=np.pi
    else:angle=[0.,np.pi,np.pi/2,-np.pi/2][i]
    target=np.array([speed*np.cos(facing+angle),speed*np.sin(facing+angle),facing])
    u=np.clip(elapsed/segment['ramp_seconds'],0,1);u=u*u*(3-2*u)
    target[:2]=previous_command[:2]*(1-u)+target[:2]*u
    return target


def motion_feature(root,q):
    """Canonical trajectory feature; no global-position/initial-heading duplicates."""
    if root.shape not in [(301,13),(601,13)] or q.shape!=(len(root),12) or not np.isfinite(root).all() or not np.isfinite(q).all():
        raise ValueError('expected finite 301- or 601-state trace')
    w=root[:,6];x=root[:,3];y=root[:,4];z=root[:,5]
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    c,s=np.cos(yaw[0]),np.sin(yaw[0]);xy=root[:,:2]-root[0,:2]
    xy=xy@np.array([[c,-s],[s,c]])
    samples=np.linspace(0,len(root)-1,32,dtype=int)
    return np.c_[xy/.5,root[:,2]/.5,np.cos(yaw-yaw[0]),np.sin(yaw-yaw[0]),q/np.pi][samples].reshape(-1).astype(np.float32)


def quality(plan,root,q):
    """Screen actual segment behavior, not merely requested command coverage."""
    frames=plan['segments'][-1]['end']
    if root.shape!=(frames+1,13) or q.shape!=(frames+1,12) or not np.isfinite(root).all() or not np.isfinite(q).all():
        return False,'incomplete_or_nonfinite',[]
    upright=1-2*(root[:,3]**2+root[:,4]**2)
    if upright.min()<.35 or root[:,2].min()<.055:return False,'fall',[]
    yaw=np.unwrap(np.arctan2(2*(root[:,6]*root[:,5]+root[:,3]*root[:,4]),1-2*(root[:,4]**2+root[:,5]**2)))
    checks=[]
    for s in plan['segments']:
        a,b=s['start'],s['end'];i=s['index'];delta=root[b,:2]-root[a,:2]
        local=delta@np.array([[np.cos(yaw[a]),-np.sin(yaw[a])],[np.sin(yaw[a]),np.cos(yaw[a])]])
        dyaw=yaw[b]-yaw[a]
        if i<4:
            direction=np.array([[1,0],[-1,0],[0,1],[0,-1]])[i]
            good=float(local@direction)>.015
        elif i in (4,5):good=dyaw*(-1 if i==4 else 1)>.04
        elif i<10:
            mid=a+int((b-a)*.4);v=root[b,:2]-root[mid,:2]
            projection=v@np.array([np.cos(yaw[mid]),np.sin(yaw[mid])])
            good=dyaw*(1 if i in (6,8) else -1)>.03 and projection*(-1 if i in (8,9) else 1)>.01
        elif i==10:good=root[b,2]<root[a,2]-.025 and root[b,2]<.23
        else:good=root[b,2]>root[a,2]+.025 and root[b,2]>.22
        # Compare against one second earlier; endpoint displacement alone admits long idle tails.
        lag=min(FPS,b-a)
        position=np.linalg.norm(root[a+lag:b+1,:2]-root[a:b+1-lag,:2],axis=1)
        heading=np.abs(yaw[a+lag:b+1]-yaw[a:b+1-lag])
        height=np.abs(root[a+lag:b+1,2]-root[a:b+1-lag,2])
        active=(position>=.03)|(heading>=np.deg2rad(5))|(height>=.02)
        idle=longest=0
        for moving in active:
            idle=0 if moving else idle+1
            longest=max(longest,idle)
        temporal=bool(active.mean()>=.5 and longest/FPS<=1.5)
        good=good and temporal
        checks.append(dict(index=i,passed=bool(good),delta_yaw=float(dyaw),height_end=float(root[b,2]),
                           active_fraction=float(active.mean()),longest_inactive_seconds=longest/FPS,
                           max_position_change_m=float(position.max()),max_heading_change_deg=float(np.rad2deg(heading.max())),
                           max_height_change_m=float(height.max()),temporal_passed=temporal))
    return all(c['passed'] for c in checks),'pass' if all(c['passed'] for c in checks) else 'segment_not_performed',checks


def make_report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    out=Path(out);plans=json.loads((out/'plan.json').read_text())['motions']
    records=[json.loads(s) for s in (out/'records.jsonl').read_text().splitlines()] if (out/'records.jsonl').exists() else []
    accepted={r['id']:r for r in records if r['accepted']}
    max_frames=max(p['segments'][-1]['end'] for p in plans)
    planned=np.full((len(plans),max_frames),-2,dtype=np.int16)
    for i,p in enumerate(plans):planned[i,:len(labels(p))]=labels(p)
    actual=np.zeros(12,dtype=int)
    for r in accepted.values():actual+=np.bincount(labels(r['plan']),minlength=12)
    colors=plt.get_cmap('tab20')(np.array([0,2,4,6,8,10,12,14,16,18,1,5]))
    fig,ax=plt.subplots(figsize=(13,4));x=np.arange(12)
    ax.bar(x-.2,np.bincount(planned[planned>=0],minlength=12)/(planned>=0).sum()*100,.4,color=colors,alpha=.35,label='planned')
    ax.bar(x+.2,actual/max(1,actual.sum())*100,.4,color=colors,label='accepted segment time')
    ax.axhline(100/12,ls='--',color='black',label='equal-time target')
    ax.set_xticks(x);ax.set_xticklabels(NAMES,rotation=35,ha='right');ax.set_ylabel('Time share (%)');ax.legend();fig.tight_layout();fig.savefig(out/'motion_shares.png',dpi=160);plt.close(fig)
    pk=np.bincount([p['k'] for p in plans],minlength=13)[1:]
    ak=np.bincount([r['plan']['k'] for r in accepted.values()],minlength=13)[1:]
    fig,ax=plt.subplots(figsize=(11,4));x=np.arange(1,13);ax.bar(x-.2,pk,.4,label='planned');ax.bar(x+.2,ak,.4,label='accepted');ax.set_xticks(x);ax.set_xlabel('K: number of unique motion indices');ax.set_ylabel('Sequences');ax.legend();fig.tight_layout();fig.savefig(out/'k_distribution.png',dpi=160);plt.close(fig)
    order=sorted(range(len(plans)),key=lambda i:(plans[i]['k'],i));matrix=np.where(planned==-2,-2,-1).astype(np.int16)
    for i,r in accepted.items():matrix[i,:len(labels(r['plan']))]=labels(r['plan'])
    fig,axes=plt.subplots(1,2,figsize=(15,12),sharey=True)
    cmap=ListedColormap(np.vstack([[1,1,1,1],[.85,.85,.85,1],colors]))
    for ax,data,title in zip(axes,[planned,matrix],['Planned command composition','Accepted physical rollouts (gray = not collected)']):
        ax.imshow(data[order]+2,cmap=cmap,vmin=0,vmax=13,aspect='auto',interpolation='nearest',extent=[0,max_frames/FPS,len(plans),0]);ax.set_title(title);ax.set_xlabel('Time (s)')
        for k in range(1,12):ax.axhline(sum(p['k']<=k for p in plans),color='white',lw=.5)
    axes[0].set_ylabel('Sequences grouped by K');fig.legend(handles=[Patch(color=colors[i],label=str(i)+' '+NAMES[i]) for i in range(12)],loc='lower center',ncol=4,fontsize=8);fig.tight_layout(rect=(0,.07,1,1));fig.savefig(out/'timeline.png',dpi=160);plt.close(fig)
    np.savez_compressed(out/'timeline_labels.npz',planned=planned,accepted=matrix,row_order=order,dt=1/FPS)
    counts=dict(target=len(plans),accepted=len(accepted),attempts=len(records),failed=sum(not r['accepted'] and r['reason']!='near_duplicate' for r in records),duplicates=sum(r['reason']=='near_duplicate' for r in records))
    (out/'coverage.json').write_text(json.dumps(counts,indent=2))
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>CALM 10,000 motion collection</title><style>body{font:16px/1.8 system-ui;max-width:1300px;margin:35px auto;padding:20px}img{width:100%}</style><h1>CALM 10·20초 연속 모션 수집</h1>'
    page+='<p>목표 {target:,} · 실제 통과 {accepted:,} · 시도 {attempts:,} · 실패 {failed:,} · 근접 중복 제외 {duplicates:,}</p>'.format(**counts)
    page+='<p>계획과 실제 수집을 분리했다. accepted 비중은 실제 수행 검사를 통과한 모션의 구간 라벨 시간이며 전환 시간을 포함한다. 프레임별 동작 인식 결과는 아니다. K&lt;6은 10초, K≥6은 20초다. 회색은 미수집, 흰색은 10초 모션 종료 뒤의 영역이다. 시간 지도는 전체를 축소 표시하며 원래 가변 길이 라벨은 timeline_labels.npz에 저장했다.</p>'
    for title,name in [('동작별 비중','motion_shares.png'),('동작 개수 K 분포','k_distribution.png'),('시간 구성 지도','timeline.png')]:page+='<h2>'+title+'</h2><img src="'+name+'">'
    page+='<p>유효 동작은 1초 전 대비 월드 XY 위치 3cm 이상, 바라보는 각도 5도 이상, 몸통 높이 2cm 이상 중 하나의 실제 변화를 요구한다. 이동·회전 구간은 비교 시점의 50% 이상이 활동 상태이고 연속 정지는 1.5초 이하여야 한다. 앉기·일어나기도 같은 시간 변화 검사를 적용한다.</p>'
    page+='<p>앉기는 CALM LLC latent, 일어나기는 CALM HLC의 정지·수직 자세 목표를 사용한다. 고정 stand-up latent가 앉은 자세에 머물러 대체 경로를 실제 시험했다. 목표 속도 랜덤화는 이동/회전에 적용하며 자세 전환 속도를 직접 제어한다고 주장하지 않는다. 자기 충돌은 끄고 바닥 충돌을 유지한다. 통과 판정은 방향·회전·높이 변화의 선별 기준이며 정밀 추종이나 TT 재현성의 보증은 아니다.</p></html>'
    (out/'report.html').write_text(page,encoding='utf-8')
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=10000);p.add_argument('--seed',type=int,default=0);p.add_argument('--report-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    path=a.out/'plan.json'
    if not a.report_only:
        if path.exists():raise FileExistsError('plan already exists; use --report-only')
        path.write_text(json.dumps(dict(seed=a.seed,motions=make_plan(a.count,a.seed),names=NAMES,fps=FPS),indent=2))
    print(make_report(a.out))


if __name__=='__main__':main()
