"""Validate a completed CALM collection and create an offline report with five videos."""
import argparse,base64,collections,hashlib,html,json,os,re,subprocess
from pathlib import Path
import numpy as np
from calm_sequences import FPS,make_report,quality


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def validate_trace(path,plan):
    n=600 if plan['k']>=6 else 300
    if plan['segments'][-1]['end']!=n:raise ValueError('plan duration mismatch')
    with np.load(path,allow_pickle=False) as z:
        expected={'root':(n+1,13),'q':(n+1,12),'qvel':(n+1,12),'foot_pos':(n+1,4,3),
                  'foot_floor_contact':(n+1,4),'target_q':(n,12),'llc_action':(n,12),'latent':(n,64),
                  'command':(n,3),'joint_force_sensor':(n,12),'time':(n+1,),'valid_transition':(n,)}
        for key,shape in expected.items():
            a=z[key]
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'{path}: invalid {key}')
        if not z['valid_transition'].all() or not np.allclose(z['time'],np.arange(n+1)/FPS):raise ValueError('incomplete time axis')
        if str(z['quaternion_order'])!='xyzw' or str(z['velocity_frame'])!='world':raise ValueError('coordinate convention mismatch')
        ok,reason,checks=quality(plan,z['root'],z['q'])
        if not ok:raise ValueError(f'{path}: no longer passes quality: {reason}')
    return dict(frames=n,duration_seconds=n/FPS,sha256=digest(path),min_active_fraction=min(c['active_fraction'] for c in checks),max_idle_seconds=max(c['longest_inactive_seconds'] for c in checks))


def select_examples(records):
    accepted=sorted((r for r in records if r['accepted']),key=lambda r:r['id']);chosen=[]
    for k in [1,3,6,9,12]:
        rows=[r for r in accepted if r['plan']['k']==k]
        def matches(r):
            ids={s['index'] for s in r['plan']['segments']}
            return ids=={0} if k==1 else bool(ids&set(range(4,10))) if k==3 else {10,11}<=ids
        found=next((r for r in rows if matches(r)),None)
        if found is None:raise ValueError(f'no representative sample for K={k}')
        chosen.append(found)
    return chosen


def render_examples(out,records):
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco,imageio.v2 as imageio,imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    from render_calm_three_models import source_spec
    from calm_sequences import NAMES
    root=Path(__file__).resolve().parents[2];folder=out/'videos';folder.mkdir(exist_ok=True)
    spec=source_spec(root)
    spec.add_texture(name='checker',type=mujoco.mjtTexture.mjTEXTURE_2D,builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,rgb1=[.2,.3,.4],rgb2=[.1,.2,.3],width=512,height=512)
    spec.add_material(name='checker_floor',textures=['','checker'],texrepeat=[2,2],texuniform=True,reflectance=.1)
    spec.worldbody.add_geom(name='example_floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[100,100,.1],material='checker_floor',group=0)
    spec.worldbody.add_light(pos=[0,0,5],dir=[0,0,-1],diffuse=[.8,.8,.8])
    model=spec.compile();assert model.nq==19
    model.vis.global_.offwidth=960;model.vis.global_.offheight=540
    options=mujoco.MjvOption();options.geomgroup[5]=False
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=540,width=960)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#aec7e8','#98df8a']
    rows=[]
    try:
        for r in select_examples(records):
            plan=r['plan'];src=out/r['file'];n=plan['segments'][-1]['end'];seconds=n/FPS
            with np.load(src) as z:
                states=z['root'];q=z['q'];addresses=[int(model.joint(str(name)).qposadr[0]) for name in z['joint_names']]
            assert len(set(addresses))==12
            path=folder/('motion_%05d_K%02d.mp4'%(r['id'],plan['k']));tmp=path.with_name(path.stem+'.tmp.mp4')
            camera=mujoco.MjvCamera();camera.azimuth=130;camera.elevation=-35;camera.distance=2.6
            with imageio.get_writer(tmp,fps=FPS,codec='libx264',quality=7,macro_block_size=1) as writer:
                for i in range(n):
                    data.qpos[:3]=states[i,:3];data.qpos[3:7]=states[i,[6,3,4,5]];data.qpos[addresses]=q[i]
                    mujoco.mj_forward(model,data);camera.lookat[:]=[states[i,0],states[i,1],.20]
                    renderer.update_scene(data,camera=camera,scene_option=options)
                    frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                    segment=next(s for s in plan['segments'] if s['start']<=i<s['end'])
                    draw.rectangle((0,0,960,73),fill='#16202b')
                    draw.text((18,8),f"CALM #{r['id']:05d} | K={plan['k']} | {i/FPS:.2f} / {seconds:.2f} s",font=font,fill='white')
                    draw.text((18,39),f"{segment['index']:02d}: {NAMES[segment['index']]} | Checker floor | Recorded physics, 1x speed",font=small,fill='white')
                    draw.rectangle((0,493,960,540),fill='#16202b')
                    for s in plan['segments']:
                        x=18+s['start']/n*924;end=18+s['end']/n*924
                        draw.rectangle((x,505,end,528),fill=colors[s['index']]);draw.text((x+2,507),str(s['index']),font=small,fill='black')
                    x=18+i/n*924;draw.line((x,499,x,534),fill='white',width=3)
                    writer.append_data(np.asarray(frame))
                    if i==n//2:imageio.imwrite(path.with_suffix('.png'),np.asarray(frame))
            tmp.replace(path);webm=path.with_suffix('.webm');tempwebm=webm.with_name(webm.stem+'.tmp.webm')
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(path),'-an','-c:v','libvpx','-b:v','1200k','-crf','10','-pix_fmt','yuv420p','-threads','2',str(tempwebm)],check=True)
            tempwebm.replace(webm)
            for video in [path,webm]:
                with imageio.get_reader(video) as reader:
                    meta=reader.get_meta_data()
                    assert reader.count_frames()==n and abs(meta['duration']-seconds)<.03 and meta['fps']==FPS
            rows.append(dict(id=r['id'],k=plan['k'],source=str(src),source_sha256=digest(src),video=str(path),webm=str(webm),poster=str(path.with_suffix('.png')),frames=n,fps=FPS,duration_seconds=seconds,segments=plan['segments']))
            save(out/'report_status.json',dict(status='rendering',videos_complete=len(rows),videos_target=5));print(path,flush=True)
    finally:renderer.close()
    return rows


def uri(path,mime):return 'data:'+mime+';base64,'+base64.b64encode(Path(path).read_bytes()).decode('ascii')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    status=json.loads((out/'status.json').read_text())
    if status.get('status')!='complete' or status['accepted']!=10000:raise ValueError('requires 10,000 completed accepted motions')
    ready=out/'report_ready.json'
    if ready.exists():ready.unlink()
    records=[json.loads(s) for s in (out/'records.jsonl').read_text().splitlines()];accepted=[r for r in records if r['accepted']]
    if len(accepted)!=10000 or len({r['id'] for r in accepted})!=10000:raise ValueError('journal must contain exactly 10,000 unique accepted motions')
    source_config=json.loads((out/'collection_config.json').read_text())
    for name,sha in source_config['source_sha256'].items():
        if digest(Path(__file__).with_name(name))!=sha:raise ValueError('collection source changed; cannot revalidate with different rules')
    audit={};save(out/'report_status.json',dict(status='validating',verified=0,target=10000))
    for j,r in enumerate(accepted,1):
        audit[str(r['id'])]=validate_trace(out/r['file'],r['plan'])
        if j%1000==0:
            save(out/'report_status.json',dict(status='validating',verified=j,target=10000));print('Validated',j,flush=True)
    counts=make_report(out);lengths=collections.Counter(v['duration_seconds'] for v in audit.values());seconds=sum(v['duration_seconds'] for v in audit.values())
    summary=dict(counts,accepted_fraction=len(accepted)/len(records),duration_counts=dict(lengths),total_seconds=seconds,total_hours=seconds/3600,
                 verified_files=len(audit),minimum_segment_active_fraction=min(v['min_active_fraction'] for v in audit.values()),maximum_inactive_seconds=max(v['max_idle_seconds'] for v in audit.values()),
                 rejection_reasons=dict(collections.Counter(r['reason'] for r in records if not r['accepted'])),files=audit)
    save(out/'validation.json',summary);rows=render_examples(out,records);save(out/'video_examples.json',dict(examples=rows,selection='Lowest accepted ID at K=1/3/6/9/12 with forward / turn / sit+stand constraints; no reward ranking.',rendering='Saved PhysX states replayed on original CALM A1 visuals; no new physical simulation.'))
    page=(out/'report.html').read_text();ko=['전진','후진','좌이동','우이동','CW 회전','CCW 회전','좌회전 후 직진','우회전 후 직진','좌회전 후 후진','우회전 후 후진','앉기','일어나기']
    intro=f'<h2>최종 수집 결과 및 검증</h2><p>완료: 실제 통과 10,000개 / 총 {len(records):,}시도. 통과율 {len(accepted)/len(records)*100:.2f}%. 총 {seconds:,.0f}초 ({seconds/3600:.2f}시간). 10초 모션 {lengths[10.]:,}개, 20초 모션 {lengths[20.]:,}개.</p><p>10,000개 파일의 상태·행동 길이, 시간축, 유한값, 좌표계와 구간별 실제 변화 기준을 모두 다시 검사했다. 파일별 해시와 검증 수치는 <a href="validation.json">validation.json</a>에 기록했다.</p>'
    page=page.replace('<h1>CALM 10·20초 연속 모션 수집</h1>','<h1>CALM 10,000개 모션 수집 완료 보고서</h1>'+intro)
    for name in ['motion_shares.png','k_distribution.png','timeline.png']:page=page.replace('src="'+name+'"','src="'+uri(out/name,'image/png')+'"')
    section='<section id="video-examples"><style>.examples{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}.examples video{width:100%}.examples article{border:1px solid #ddd;padding:16px;border-radius:10px}</style><h2>대표 모션 영상 5개</h2><p>K=1·3·6·9·12의 실제 통과 모션이다. K&lt;6은10초, K≥6은20초, 모두1배속이다. 영상은 저장된 물리 상태를 원본CALM A1 시각 모델에 재생했다. MuJoCo 체크무늬 바닥, 고정 시선 각도의 추적 카메라를 사용한다. WebM·MP4와 포스터를HTML에 내장해 로컬 경로 접근 없이 재생한다. 선택은 동작 구성에 따른 예시이며 보상 순위가 아니다.</p><div class="examples">'
    for row in rows:
        sequence=' → '.join(ko[s['index']] for s in row['segments'])
        section+=f'<article><h3>#{row["id"]:05d} · K={row["k"]} · {row["duration_seconds"]:.0f}초</h3><video controls preload="none" playsinline poster="{uri(row["poster"],"image/png")}"><source src="{uri(row["webm"],"video/webm")}" type="video/webm"><source src="{uri(row["video"],"video/mp4")}" type="video/mp4"></video><p>{html.escape(sequence)}</p><p><a href="{html.escape(row["video"],quote=True)}">영상 절대 경로</a> · <a href="{html.escape(row["source"],quote=True)}">원본 데이터</a></p></article>'
    section+='</div><p>대표 영상은 전체 다양성의 통계 검증을 대신하지 않는다. 시간 변화 검사는 정밀 명령 추종·발 미끄러짐·TT 리타겟팅 성능을 보증하지 않는다. 자기 충돌을 끈 CALM PhysX 데이터다.</p></section>'
    page=page.replace('</html>',section+'</html>');tmp=out/'report.html.tmp';tmp.write_text(page);tmp.replace(out/'report.html')
    save(ready,dict(ready=True,accepted=10000,video_count=5,source_config_sha256=digest(out/'collection_config.json'),validation_sha256=digest(out/'validation.json'),report_sha256=digest(out/'report.html')))
    save(out/'report_status.json',dict(status='complete',verified=10000,videos_complete=5));print('Final report ready',flush=True)

if __name__=='__main__':main()
