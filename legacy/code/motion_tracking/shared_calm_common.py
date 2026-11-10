"""Portable state files and deterministic all-motion assignments for shared PPO."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from plan_calm_dynamics import physical_dimensions,length_ranges,LENGTH_TOKENS

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError):return default

def event(out,kind,**values):
    with (Path(out)/'events.jsonl').open('a') as f:f.write(json.dumps({**values,'time':time.time(),'kind':kind},ensure_ascii=False)+'\n')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def shapes(seed):
    rng=np.random.default_rng(seed);active=list(LENGTH_TOKENS);changes=[{}]
    for i in active:
        for d in [-.20,-.15,-.10,-.05,.05,.10,.15,.20]:changes.append({i:d})
    import itertools
    for i,j in itertools.combinations(active,2):
        for a,b in itertools.product([-.2,.2],repeat=2):changes.append({i:a,j:b})
    for ds in itertools.product([-.2,.2],repeat=5):changes.append(dict(zip(active,ds)))
    for k in [3,4,5]:
        for _ in range(29):changes.append(dict(zip(rng.choice(active,k,replace=False),rng.uniform(-.19,.19,k))))
    assert len(changes)==200
    for _ in range(20):changes.append(dict(zip(active,rng.uniform(-.195,.195,5))))
    ranges=length_ranges();rows=[]
    for n,deltas in enumerate(changes):
        e=np.full(13,.5)
        for i,d in deltas.items():
            lo,hi=ranges[i];base=(lo+hi)/2;e[i]=(base*(1+d)-lo)/(hi-lo)
        rows.append(dict(design=f'e{n:04d}',E=e.tolist(),physical_dimensions=physical_dimensions(e),changed_tokens=[int(i) for i in deltas]))
    assert len({tuple(r['E']) for r in rows})==220
    return rows

def create_plan(source,config):
    source=Path(source).resolve();c=config
    if read(source/'status.json',{}).get('accepted')!=10000:raise ValueError('Source dataset must contain 10,000 accepted motions')
    rows=[r for r in map(json.loads,(source/'records.jsonl').read_text().splitlines()) if r['accepted']]
    rows=sorted(rows,key=lambda r:r['id'])[:c['source_motions']]
    if len(rows)!=c['source_motions'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Missing/duplicate source motions')
    n=c['train_designs'];repeats=c['morphologies_per_motion'];count=len(rows)*repeats
    if not 1<repeats<=n or count%n or n%repeats:raise ValueError('Balanced assignment requires source*reuse divisible by designs')
    per=count//n
    if len(rows)%per:raise ValueError('Source size must divide clip slots exactly')
    rng=np.random.default_rng(c['seed']);rng.shuffle(rows)
    catalog=shapes(c['seed']);train=catalog[:n];validation=catalog[200:200+c['eval_designs']]
    if len(train)!=n or len(validation)!=c['eval_designs']:raise ValueError('Up to 200 train and 20 validation designs supported')
    assignments={e['design']:[] for e in train}
    for i,r in enumerate(rows):
        for j in range(repeats):assignments[train[(i//per+j*(n//repeats))%n]['design']].append(r['id'])
    if any(len(set(v))!=per for v in assignments.values()):raise ValueError('Duplicate source within design')
    for e in validation:assignments[e['design']]=[r['id'] for r in rng.choice(rows,per,replace=False)]
    motions=[dict(id=r['id'],k=r['plan']['k'],file=str(source/r['file']),sha256=sha(source/r['file']),seconds=r['plan']['segments'][-1]['end']/30,segments=r['plan']['segments']) for r in rows]
    return dict(version=1,config=c,topology='00000011',source=str(source),motions=motions,designs=train,eval_designs=validation,assignments=assignments,
                target_pairs=count,target_episodes=count*c['rollouts_per_pair'],clips_per_design=per)

def stage_groups(plan):
    names=[r['design'] for r in plan['designs']];n=plan['config']['resident_designs']
    return [names[i:i+n] for i in range(0,len(names),n)]

def load_config(path):
    c=OmegaConf.to_container(OmegaConf.load(path),resolve=True)
    if c['hours']<=0 or not math.isclose(sum(c['phase_hours'].values()),c['hours']) or any(v<=0 for v in c['phase_hours'].values()):raise ValueError('Phase budgets must sum to positive total hours')
    for key in ['num_envs','batch_size','num_minibatches','unroll_length','resident_designs','rollout_batch','train_transitions','train_sweeps','prepare_workers']:
        if not isinstance(c[key],int) or c[key]<1:raise ValueError('Positive integer required: '+key)
    if not 0<c['pool_fraction']<1:raise ValueError('Memory pool fraction must be between zero and one')
    if c['physics_hz']!=8*c['control_hz'] or c['control_hz']!=30:raise ValueError('Expected 30Hz control / 240Hz physics')
    if c['batch_size']*c['num_minibatches']%c['num_envs']:raise ValueError('Invalid PPO batch divisibility')
    if c.get('action_reference','home') not in ('home','reference'):raise ValueError('Invalid action reference')
    if not math.isfinite(c.get('init_noise_std',1.)) or c.get('init_noise_std',1.)<=0:raise ValueError('Positive finite exploration std required')
    if c['rollouts_per_pair']!=2:raise ValueError('Two distinct starts are implemented')
    return c


def ready_batch(out,plan,index):
    ready=Path(out)/'ready'
    names=[n for n in stage_groups(plan)[index] if (ready/(n+'.json')).exists()]
    heldout=[r['design'] for r in plan['eval_designs'] if (ready/(r['design']+'.json')).exists()]
    return names,([heldout[index%len(heldout)]] if heldout else [])


def covered_length(intervals):
    end=total=0
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end));end=max(end,b)
    return total


def choose_undercovered_start(motion,intervals,rng):
    segments=motion.get('segments') or [{'start':0,'end':round(motion['seconds']*30),'index':-1}]
    coverage=[]
    for s in segments:
        overlap=[(max(a,s['start']),min(b,s['end'])) for a,b in intervals if b>s['start'] and a<s['end']]
        coverage.append(covered_length(overlap)/max(1,s['end']-s['start']))
    candidates=np.flatnonzero(np.isclose(coverage,min(coverage)))
    s=segments[int(rng.choice(candidates))];latest=max(s['start']+1,(s['start']+s['end'])//2)
    return int(rng.integers(s['start'],latest))
