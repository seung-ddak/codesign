"""Build an evidence-based Korean G0 report; downstream gates never auto-pass."""
import argparse
import html
import json
import os
from pathlib import Path
import numpy as np


def gate_status(summary):
    required = ('continuous_10s', 'finite_states', 'finite_actions')
    if any(summary.get(k) is False for k in required) or summary.get('resets_after_initial',0)>0:
        return 'FAIL'
    # No numeric command-fidelity approval has been established yet.
    return 'BLOCKED'


def validate_arrays(data):
    state, action = data['state'], data['llc_action']
    if state.ndim!=2 or state.shape[1]!=37 or action.shape!=(len(state)-1,12):
        raise ValueError('State/action shape or time alignment mismatch')
    t=data['time'];dt=float(data['dt'])
    if len(t)!=len(state) or not np.allclose(np.diff(t),dt,rtol=1e-5):
        raise ValueError('Timestamp mismatch')
    return bool(all(np.isfinite(data[k]).all() for k in ['state','llc_action','target_q','latent','command']))


def report(episode,out):
    episode,out=Path(episode).resolve(),Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    summary=json.loads((episode/'summary.json').read_text())
    manifest=json.loads((episode/'manifest.json').read_text())
    with np.load(episode/'episode.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    valid=validate_arrays(data)
    if not valid: summary['finite_states']=False
    status=gate_status(summary)
    stages={f'G{i}':{'status':status if i==0 else 'BLOCKED',
                    'executed':1 if i==0 else 0,
                    'reason':'연속 실행과 명령 추종 검증 실패' if i==0 else '선행 단계 G0 미통과'} for i in range(9)}
    result=dict(stages=stages,executed_g0=1,passed_g0=0,failed_g0=int(status=='FAIL'),
                blocked_g0=int(status=='BLOCKED'),array_integrity='PASS' if valid else 'FAIL',
                measured=summary,execution_manifest=manifest)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Noto Sans CJK KR';plt.rcParams['axes.unicode_minus']=False
    state=data['state'];t=data['time'];cmd=data['command'];q=state[:,3:7]
    yaw=np.unwrap(np.arctan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)))
    f,ax=plt.subplots(3,2,figsize=(14,11))
    ax[0,0].plot(state[:,0],state[:,1],'.-');ax[0,0].set(title='실제 몸통 이동 궤적',xlabel='앞뒤 위치 (m)',ylabel='좌우 위치 (m)')
    for j,label in enumerate(['앞뒤','좌우']):
        ax[0,1].plot(t[1:],state[1:,7+j],label='실제 '+label)
        ax[0,1].plot(t[:-1],cmd[:,j],'--',label='명령 '+label)
    ax[0,1].set(title='명령 속도와 실제 속도',ylabel='속도 (m/s)');ax[0,1].legend()
    ax[1,0].plot(t,np.degrees(yaw),label='실제 방향');ax[1,0].plot(t[:-1],np.degrees(cmd[:,2]),'--',label='목표 방향')
    ax[1,0].set(title='목표 방향과 실제 방향',ylabel='방향 각도 (도)');ax[1,0].legend()
    ax[1,1].plot(t,state[:,13:25]);ax[1,1].set(title='실제 관절각 12개',ylabel='관절각 (rad)')
    ax[2,0].plot(t[:-1],data['torque']);ax[2,0].set(title='관절 힘 센서값 12개',ylabel='센서값 (Nm)')
    ax[2,1].plot(t[:-1],data['target_q']);ax[2,1].set(title='실제 적용한 목표 관절각 12개',ylabel='목표각 (rad)')
    for a in ax.flat:a.grid(alpha=.25)
    for a in ax[1:,:].flat:a.set_xlabel('시간 (초)')
    f.tight_layout();f.savefig(out/'measurements.png',dpi=140);plt.close(f)
    rel=html.escape(os.path.relpath(episode,out),quote=True)
    diagnostic_notes=[]
    for name in ['llc_first_original','llc_first_warmup','llc_axes_control_audit_try2']:
        path=episode.parent/name/'summary.json'
        if path.exists():
            measured=json.loads(path.read_text())
            terminated=sum(step>=0 for step in measured['first_done_steps'])
            count=len(measured['first_done_steps'])
            error=max(measured['reset_fk_max_error_m'])
            href=html.escape(os.path.relpath(path,out),quote=True)
            diagnostic_notes.append(f'<p><a href="{href}">{name}</a>: {count}개 중 {terminated}개 종료, '
                                    f'{count-terminated}개 미종료. 초기 관측 위치 오차 최대 {error:.6f}m.</p>')
    diagnostic_html=''.join(diagnostic_notes) or '<p>별도 진단 자료 없음: NOT RUN</p>'
    axis=episode.parent/'llc_axes_control_audit_try2/summary.json'
    if axis.exists():
        a=json.loads(axis.read_text())
        ups=[u for k,u in zip(a['knee_termination'],a['terminal_body_up']) if k and u is not None]
        diagnostic_html+=f'<p>원본 URDF와 실제 강체의 위치 차이 최대 {a["full_body_fk_position_error_m"]:.8f}m, 회전 Quaternion 비교 오차 {a["full_body_fk_quaternion_error"]:.8f}. '
        diagnostic_html+=f'낮은 무릎으로 종료된 {len(ups)}개의 몸통 위쪽 성분은 {min(ups):.3f}~{max(ups):.3f}로 모두 위를 향했습니다. 이 종료를 모두 전복으로 부르면 부정확합니다.</p>'
    probe=episode.parent/'llc_alternate_reset_probe/reset_probe.json'
    if probe.exists():
        p=json.loads(probe.read_text())
        changed=sum(a!=b for a,b in zip(p['first']['motion_ids'],p['second']['motion_ids']))
        tilted=sum(u<0 for u in p['second']['body_up'])
        diagnostic_html+=f'<p>원본 기본 초기화 방식을 두 번 호출한 결과 {len(p["first"]["motion_ids"])}개 중 {changed}개가 자기 모션과 다른 모션의 자세를 받았고, {tilted}개는 몸통 위쪽 축이 아래를 향했습니다. '
        diagnostic_html+='정책이나 물리 스텝 없이 발생한 자세 변경입니다. 화면상 뒤집힘에는 이 초기화 동작이 섞일 수 있습니다. G0 성공을 뜻하지 않습니다.</p>'

    rows=''.join(f'<tr><td>{g}</td><td>{v["status"]}</td><td>{v["executed"]}</td><td>{v["reason"]}</td></tr>' for g,v in stages.items())
    info=html.escape(json.dumps(manifest,ensure_ascii=False,indent=2))
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>G0 실행 검증</title>
<style>body{{font:17px system-ui;max-width:1150px;margin:32px auto;padding:0 18px;line-height:1.65}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:9px;text-align:left}}video,img{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:16px}}.fail{{color:#b91c1c}}</style>
<h1>G0 원본 A1 실행 검증: <span class="fail">{status}</span></h1>
<p>실행 1회 · 통과 0회 · 실패 {int(status=='FAIL')}회 · 진행 불가 {int(status=='BLOCKED')}회. 목표는 연속 10초와 정상적인 명령 반응입니다.</p>
<table><tr><th>항목</th><th>측정 결과</th></tr>
<tr><td>실제 실행 길이</td><td>{summary['seconds']:.3f}초 / 목표 10초 · {summary['steps']}스텝</td></tr>
<tr><td>명령 속도와 실제 속도의 평균 제곱근 오차</td><td>{summary['command_velocity_rmse_mps']:.3f}m/s</td></tr>
<tr><td>중간 초기화</td><td>{summary['resets_after_initial']}회</td></tr>
<tr><td>상태·행동 배열 시간 대응</td><td>{result['array_integrity']} · 상태 {len(state)}개, 행동 {len(state)-1}개</td></tr>
<tr><td>유한한 상태·목표각·행동·잠재벡터</td><td>{'PASS' if valid else 'FAIL'}</td></tr>
<tr><td>초기 관절 상태와 강체 관측의 무릎 위치 차이</td><td>{summary['reset_fk_error_m']:.6f}m</td></tr>
<tr><td>실행 설정</td><td>원본 A1 · Isaac Gym · hit2/31zs · seed {manifest['seed']} · 초기 관측 보정 {manifest['sync_reset']}</td></tr>
<tr><td>실행 당시 커밋</td><td>{html.escape(manifest['git_commit'])} · 미커밋 소스는 아래 파일별 해시로 식별</td></tr></table>
<h2>실제 실행 영상과 측정값</h2><p>실제로 기록된 {summary['seconds']:.3f}초 영상입니다. 10초 영상이나 성공 사례로 처리하지 않습니다. 회전·횡이동·정지 구간은 도달하지 못했습니다.</p>
<video controls src="{rel}/rollout.mp4"></video><img src="measurements.png" alt="실제 궤적, 속도, 방향, 관절각, 힘 센서값과 목표각 그래프">
<p>힘 센서값은 순수한 구동 토크라고 검증되지 않았습니다. 초기 접촉과 몸체 전체 관통 여부도 아직 통과 판정하지 않았습니다.</p>
<h2>종료 진단</h2><pre>{html.escape(json.dumps(summary.get('terminal_diagnostics',{}),indent=2))}</pre>
<p>확인된 직접 종료 조건은 낮은 무릎 상태 3프레임입니다. 이것은 종료를 발생시킨 조건이며, 정책이 왜 그 상태로 갔는지에 대한 최종 원인은 아닙니다.</p>
<h2>함께 수행한 원인 분리 실험</h2>
{diagnostic_html}
<p>진단은 각 모션의 첫 프레임으로 시작하고, 최초 종료 시점만 집계했습니다. 종료되지 않았다는 이유로 정상 모션으로 판정하지 않았습니다. 준비 스텝은 별도 진단에만 적용했고 G0 원본 실행에는 적용하지 않았습니다.</p>
<p>관측 보정만 적용한 이전 비교 시도 2회는 원본 낙상 초기 상태 생성 중 CUDA 오류로 종료됐습니다. 정책 비교 결과가 없는 실행은 진행 불가로 구분합니다. 이전 진단 기록에는 현재 기록 체계의 완전한 소스 해시가 없어 재현성에 제한이 있습니다.</p>
<h2>확인되지 않은 원인 가설과 다음 작업</h2>
<p>좌표축·관절 순서·정규화·행동 변환·학습 당시 물리 설정과의 불일치는 아직 원인으로 확정되지 않았습니다. 현재 실패는 MJX 변환 이전의 원본 Isaac Gym에서 발생합니다.</p>
<p>다음 최소 검증은 제공자의 정상 실행 환경과 패키지 버전·관측 순서·행동 변환을 대조하는 것입니다. 종료 조건이나 제어 이득을 바꿔 통과시키지 않습니다.</p>
<h2>전체 단계 상태</h2><table><tr><th>단계</th><th>상태</th><th>이번 실행 수</th><th>이유</th></tr>{rows}</table>
<p>G1~G8의 새 수집·선정·리타겟팅·학습은 실행하지 않았습니다. 이전 V7 결과를 새 파이프라인의 통과 증거로 사용하지 않습니다.</p>
<h2>원본 자료</h2><p><a href="{rel}/episode.npz">상태·행동 데이터</a> · <a href="{rel}/summary.json">실행 요약</a> · <a href="{rel}/manifest.json">설정·체크포인트 해시</a> · <a href="{rel}/diagnostics.json">종료 진단</a> · <a href="{rel}.log">실행 로그</a> · <a href="results.json">단계별 결과</a></p>
<details><summary>실행 명령·설정·출처 전체 보기</summary><pre>{info}</pre></details></html>'''
    (out/'report.html').write_text(text)
    return result




def diagnostic_reports(root):
    """Preserve a separate result page for each recorded attempt, including errors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Noto Sans CJK KR'
    plt.rcParams['axes.unicode_minus']=False
    pages=[]
    for path in sorted(Path(root).iterdir()):
        log=path.with_suffix('.log')
        if not path.is_dir() or not log.is_file() or (path/'report.html').exists():
            continue
        summary=path/'summary.json';probe=path/'reset_probe.json'
        measured=json.loads(summary.read_text()) if summary.exists() else json.loads(probe.read_text()) if probe.exists() else {}
        if 'first_done_steps' in measured:
            first=np.asarray(measured['first_done_steps']);failed=int((first>=0).sum())
            status='FAIL' if failed else 'BLOCKED'
            description=f'정책 실행 {len(first)}개 중 {failed}개 조기 종료. 나머지는 2초 동안 미종료이며 정상 추종 성공을 뜻하지 않습니다.'
            f,ax=plt.subplots(figsize=(10,3))
            ax.bar(np.arange(len(first)),np.where(first>=0,first*measured['dt'],2),color=np.where(first>=0,'#b91c1c','#888888'))
            ax.set(xlabel='로봇 번호',ylabel='초',title='최초 종료 시점 · 회색은 2초 동안 미종료')
            f.tight_layout();f.savefig(path/'first_termination.png',dpi=140);plt.close(f)
        elif 'continuous_10s' in measured:
            status=gate_status(measured);description='HLC와 LLC 연속 실행 검증. 전체 G0 통과와 구분합니다.'
        elif probe.exists():
            status=measured['status'];description='초기화 동작만 검증한 결과이며 G0나 정책 성공 판정이 아닙니다.'
        else:
            status='BLOCKED';description='결과 요약을 만들기 전에 실행 오류가 발생했습니다. 정책 성능 판정에 사용하지 않습니다.'
        video='<video controls src="rollout.mp4"></video>' if (path/'rollout.mp4').exists() else '<p>실제 영상: NOT RUN · 이 진단에서 기록하지 않았습니다.</p>'
        graph='<img src="first_termination.png" alt="로봇별 최초 종료 시점">' if (path/'first_termination.png').exists() else ''
        body=f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>G0 진단</title><style>body{{max-width:1050px;margin:30px auto;font:17px system-ui;padding:15px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}video,img{{width:100%}}</style><h1>G0 진단: {html.escape(path.name)} · {status}</h1><p>실행 시도 1회. {description}</p>{video}{graph}<p><a href="../{html.escape(log.name)}">원본 로그</a></p><pre>{html.escape(json.dumps(measured,ensure_ascii=False,indent=2))}</pre></html>'
        with (path/'report.html').open('x') as file:file.write(body)
        pages.append(dict(attempt=path.name,status=status,meaning=description))
    return pages


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();report(a.episode,a.out)
