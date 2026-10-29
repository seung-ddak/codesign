"""Four separate V7 follow-ups, with durable Issue #8 milestone notifications."""
import argparse
import base64
import html
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/motion_tracking'))
from notify_selected_experts import post, TOKEN_FILE
from eval_v7_ddim100 import analyze_fixed, run_generation
from motion_screen import screen

P = sys.executable
BASE = ROOT/'runs/pipeline_v1/fast_mode24_v7_eswap'
EXPERT = ROOT/'runs/pipeline_v1/experts_wide16/walkForwardSlow__r00'


def read(path):
    return json.loads(path.read_text())


def flush_notifications(out, issue):
    if not TOKEN_FILE.is_file() or TOKEN_FILE.stat().st_mode & 0o077:
        raise RuntimeError('GitHub notification token missing or not private (mode 600 required)')
    for path in sorted((out/'events').glob('*.json')):
        event=read(path)
        if event.get('comment_id'):
            continue
        for attempt in range(3):
            try:
                response=post(f'/issues/{issue}/comments', {'body':event['comment']}, TOKEN_FILE.read_text().strip())
                event['comment_id']=response['id'];event.pop('notification_error',None)
                path.write_text(json.dumps(event,ensure_ascii=False,indent=2));break
            except Exception as exc:
                event['notification_error']=str(exc)
                path.write_text(json.dumps(event,ensure_ascii=False,indent=2))
                if attempt<2: time.sleep(2**attempt)
        if not event.get('comment_id'):
            raise RuntimeError(f'GitHub comment delivery failed; event retained at {path}')


def record(out, issue, title, data):
    events=out/'events';events.mkdir(exist_ok=True)
    event=dict(title=title,data=data,comment_id=None,
               comment=f'@seung-ddak **{title}**\n\n```json\n{json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)}\n```\n\n실험: `{out.relative_to(ROOT)}` · 5 diffusion steps, 기존 V7과 분리 비교입니다. 과제 기준 미달과 실행 오류는 구분합니다.')
    path=events/f'{len(list(events.glob("*.json")))+1:03d}.json'
    path.write_text(json.dumps(event,ensure_ascii=False,indent=2,allow_nan=False))
    print(title,json.dumps(data,ensure_ascii=False),flush=True)
    flush_notifications(out,issue)


def report(out):
    sections=[]
    for path in sorted(out.rglob('motion_screen.json')):
        result=read(path);rows=[]
        for scale,m in result['scales'].items():
            rates=m['rates']
            rows.append(f'<tr><td>{scale}</td><td>{rates["valid"]:.0%}</td><td>{rates["survived"]:.0%}</td><td>{rates["moving"]:.0%}</td><td>{rates["task_success"]:.0%}</td><td>{m["median_progress_m"]}</td><td>{m["median_path_error_m"]}</td><td>{m["provisional_screen_pass"]}</td></tr>')
        directory=path.parent;summary=read(directory/'summary.json')
        selected=summary['runs'].get('scale_10',[])
        cards=[]
        for label,key in [('TT 예측 보상 1위','pred_total'),('MJX 실제 보상 1위','mjx_total')]:
            choices=[i for i,r in enumerate(selected) if r[key] is not None and not r.get('execution_skipped',False)]
            if not choices:continue
            i=max(choices,key=lambda i:selected[i][key]);video=directory/f'videos/scale_10_cand{i}.mp4'
            if video.exists():
                src='data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()
                cards.append(f'<article><h3>{label} · joint {i}</h3><video controls muted loop playsinline src="{src}"></video></article>')
        detail=html.escape(json.dumps(result.get('paired_guidance',{}),ensure_ascii=False,indent=2))
        sections.append(f'<section><h2>{html.escape(str(directory.relative_to(out)))}</h2><table><tr><th>scale</th><th>유효 S0</th><th>생존</th><th>최소 이동</th><th>과제 성공</th><th>진행 m</th><th>경로 차 m</th><th>실험용 screen</th></tr>'+''.join(rows)+'</table><div class="grid">'+''.join(cards)+'</div><details><summary>같은 seed 후보의 guidance 보상 변화</summary><pre>'+detail+'</pre></details></section>')
    events=[read(p) for p in sorted((out/'events').glob('*.json'))]
    history=html.escape(json.dumps([{k:e[k] for k in ('title','data')} for e in events],ensure_ascii=False,indent=2))
    text='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>V7 분리 비교</title><style>body{font:16px/1.6 system-ui;background:#101820;color:#e5eef3;padding:24px}main{max-width:1200px;margin:auto}section,article{padding:18px;border:1px solid #354653;border-radius:10px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}video{width:100%}td,th{padding:6px;text-align:left;border-bottom:1px solid #354653}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#8ed4ed}</style><main><h1>V7 · 평가 / S guidance / 50:50 / Mode 1 분리 비교</h1>
<p>기존 V6/V7을 보존한다. Diffusion은 5단계로 유지하고 100단계 변경과 섞지 않는다. Gradient 비교는 재학습하지 않는다. 데이터 비중 및 Mode 1 비교는 각각 새로 60k 학습한다.</p>
<p>최소 이동: 0.075m/s × 평가 시간(2초 0.15m). 원형은 전진 회전 방향의 호 길이로 환산한다. 실험용 screen: 유효 초기 자세 75%·생존 75%·경로 차 중앙값 0.15m 미만·최소 이동 중앙값. 이 값은 검증된 task 성공 기준이 아니다.</p>
<p>별도 과제 성공: 지그재그는 양쪽 꺾임 지점을 순서대로 통과해 한 주기(기본 2m) 진행, 원형은 한 바퀴 진행하며, 모두 제한 시간 생존과 전체 관측 구간의 root 경로 유지를 요구한다. 경로는 50Hz root 위치 기준이며 몸체 방향 추종이나 전체 몸체 충돌 판정은 추가하지 않았다.</p>
<p>Mode 1/2/4 비중은 20/40/40%로, Mode 2·4의 학습 노출량은 기존보다 줄어든다. 원본·E-swap 검증 loss를 모드별로 별도 기록한다. 보상 정의는 변경하지 않았다.</p>'''
    (out/'report.html').write_text(text+''.join(sections)+'<section><h2>이벤트 및 학습·평가 기록</h2><pre>'+history+'</pre></section></main></html>')


def evaluate_directory(directory):
    result=screen(directory)
    (directory/'motion_screen.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return dict(task=result['task'],seconds=result['seconds'],scales={k:{key:v for key,v in m.items() if key!='candidates_metrics'} for k,m in result['scales'].items()},paired_guidance=result['paired_guidance'])


def generate(out, checkpoint, task, seconds, scales, s_weight=1., fixed=None, render=False):
    command=[P,str(ROOT/'code/tt_dynamics/generate_task.py'),'--ckpt',str(checkpoint),'--out',str(out),
        '--expert-config',str(EXPERT),'--anchor-s0','--joint-full-guidance','--e-rest-s0','--no-s0-align',
        '--seed','0','--seconds',str(seconds),'--candidates','4' if fixed else '16','--task',task,
        '--scales',*[str(s) for s in scales],'--s-gradient-scale',str(s_weight)]
    if fixed:command+=['--fixed-e',*fixed]
    if not render:command+=['--no-render']
    run_generation(command)
    return evaluate_directory(out)


def train(out, mode1=False):
    if out.exists():raise FileExistsError(f'Preserving existing training directory: {out}')
    cfg=read(BASE/'tt_joint/config.json')
    command=[P,str(ROOT/'code/tt_dynamics/train_tt.py'),'--data',*cfg['data'],'--out',str(out),
        '--horizon','32','--steps','60000','--batch','32','--lr',str(cfg['lr']),
        '--d-model',str(cfg['d_model']),'--layers',str(cfg['layers']),
        '--diff-steps','5','--norm','minmax','--clip','1','--sampler','ddim','--seed','0',
        '--eval-episodes','8','--swap-fraction','.5','--norm-from',str(BASE/'tt_joint/checkpoint.pt'),
        '--modes',*(['1','2','4'] if mode1 else ['2','4'])]
    if mode1:command+=['--mode-weights','1','2','2']
    with (out.parent/'train.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    progress=json.loads((out/'progress.jsonl').read_text().splitlines()[-1])
    assert progress['step']==60000 and (out/'checkpoint.pt').is_file()
    assert progress['sampled_sources']==[960000,960000]
    return dict(final=progress,validation_by_source_mode=read(out/'validation_by_source_mode.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--issue',type=int,default=8)
    args=p.parse_args();out=args.out.resolve();os.chdir(ROOT)
    if not out.is_relative_to(ROOT):raise ValueError('output must be under codesign')
    if (out/'events').exists():raise FileExistsError('Existing experiment retained; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'events').mkdir()
    # Validate delivery before spending GPU time; failed events remain on disk for retry.
    if not TOKEN_FILE.is_file():raise RuntimeError('GitHub notification token unavailable')
    failures=[]
    def stage(name, action):
        record(out,args.issue,name+' 시작',{'status':'running'})
        report(out)
        try:
            result=action()
        except Exception as exc:
            failures.append(name)
            record(out,args.issue,name+' 실행 오류',{'status':'execution_error','error':str(exc)})
            report(out);return False
        status='completed'
        if isinstance(result,dict) and 'scales' in result:
            status='completed_with_screen_failures' if any(not v['provisional_screen_pass'] for v in result['scales'].values()) else 'completed_screen_pass'
        if isinstance(result,dict) and result.get('morphology_gate_pass') is False:
            status='completed_with_morphology_gate_failure'
        record(out,args.issue,name+' 완료',{'status':status,'result':result})
        report(out);return True
    def terminate(signum, frame):
        record(out,args.issue,'실험 중단',{'signal':signum})
        report(out);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def reevaluate():
        results={}
        for version in ('fast_mode24_v6_steps60k','fast_mode24_v7_eswap'):
            source=BASE.parent/version
            directories=[source/'smoke_zigzag_joint_c16']
            directories += [source/f'codesign_check/{kind}_{task}_4s' for task in ('zigzag','circle') for kind in ('fixedE','joint')]
            for directory in directories:
                if not (directory/'summary.json').exists():continue
                metrics=screen(directory)
                target=out/'reevaluation'/version/directory.name;target.mkdir(parents=True)
                (target/'motion_screen.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
                # Reports need the original summary for task/candidate metadata, not modified historical metrics.
                (target/'summary.json').write_text((directory/'summary.json').read_text())
                results[f'{version}/{directory.name}']={s:{k:v for k,v in m.items() if k!='candidates_metrics'} for s,m in metrics['scales'].items()}
        return results
    stage('1. 기존 V6/V7 재평가',reevaluate)
    def gpu_self_check():
        target=out/'gpu_self_check'
        cfg=read(BASE/'tt_joint/config.json')
        original=next(p for p in cfg['data'] if '/e_swap/' not in p)
        swapped=next(p for p in cfg['data'] if '/e_swap/' in p)
        command=[P,str(ROOT/'code/tt_dynamics/train_tt.py'),'--data',original,swapped,'--out',str(target),
                 '--horizon','32','--steps','2','--batch','4','--d-model','32','--layers','1',
                 '--diff-steps','5','--norm','minmax','--clip','1','--sampler','ddim',
                 '--eval-episodes','1','--swap-fraction','.5','--norm-from',str(BASE/'tt_joint/checkpoint.pt'),
                 '--modes','1','2','4','--mode-weights','1','2','2']
        with (out/'gpu_self_check.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        last=json.loads((target/'progress.jsonl').read_text().splitlines()[-1])
        assert last['step']==2 and last['sampled_sources']==[4,4]
        validation=read(target/'validation_by_source_mode.json')
        assert set(validation['original']['modes'])=={'1','2','4'}
        return dict(status='implementation_check_only_not_task_evidence',sampled_sources=last['sampled_sources'])
    if not stage('GPU 학습 구현 2-step 검사',gpu_self_check):
        (out/'exit_code').write_text('1')
        raise SystemExit(1)
    for weight,label,scales in ((1.,'gradient_s1',[0,10]),(.3,'gradient_s03',[10])):
        for task in ('zigzag','circle'):
            stage(f'2. {label} {task}',lambda weight=weight,label=label,scales=scales,task=task:
                  generate(out/label/task,BASE/'tt_joint',task,2,scales,weight,render=True))
    def compare_gradients():
        result={}
        for task in ('zigzag','circle'):
            one=read(out/f'gradient_s1/{task}/motion_screen.json')['scales']['scale_10']
            part=read(out/f'gradient_s03/{task}/motion_screen.json')['scales']['scale_10']
            pairs=[(a,b) for a,b in zip(one['candidates_metrics'],part['candidates_metrics']) if a['valid'] and b['valid']]
            import numpy as np
            result[task]=dict(paired_valid=len(pairs),
                median_reward_delta_s03_minus_s1=float(np.median([b['reward']-a['reward'] for a,b in pairs])) if pairs else None,
                median_progress_delta_m=float(np.median([b['progress_m']-a['progress_m'] for a,b in pairs])) if pairs else None,
                rates_s1=one['rates'],rates_s03=part['rates'])
        (out/'gradient_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        return result
    stage('2. S gradient 동일 seed 비교 판정',compare_gradients)
    designs=[f'e{i:03d}' for i in range(17) if i!=3]
    for mode1,label in ((False,'balanced'),(True,'balanced_mode1')):
        variant=out/label;variant.mkdir()
        if not stage(f'{label} TT 60k 학습',lambda variant=variant,mode1=mode1:train(variant/'tt_joint',mode1)):
            continue
        stage(f'{label} 2초 smoke',lambda variant=variant:generate(variant/'smoke',variant/'tt_joint','zigzag',2,[0,10]))
        for task in ('zigzag','circle'):
            for kind in ('fixedE','joint'):
                stage(f'{label} {kind} {task}',lambda variant=variant,task=task,kind=kind:
                      generate(variant/f'codesign/{kind}_{task}_4s',variant/'tt_joint',task,4,[10],
                               fixed=designs if kind=='fixedE' else None,render=kind=='joint'))
        for task in ('zigzag','circle'):
            def compare(task=task,variant=variant):
                result=analyze_fixed(read(variant/f'codesign/fixedE_{task}_4s/summary.json'),
                                     read(variant/f'codesign/joint_{task}_4s/summary.json'),10)
                (variant/f'codesign/{task}_morphology.json').write_text(json.dumps(result,indent=2,allow_nan=False))
                return result
            stage(f'{label} {task} 형상 민감도 판정',compare)
    record(out,args.issue,'분리 비교 종료',{'execution_failures':failures,'report':str(out/'report.html'),
               'note':'실행 완료는 성능 개선이나 과제 성공을 의미하지 않습니다. 보고서의 개별 판정과 비교하세요.'})
    report(out)
    (out/'exit_code').write_text('1' if failures else '0')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
