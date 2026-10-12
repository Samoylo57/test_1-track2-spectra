"""Validate the real submission image against pinned public task prefixes.

Inputs and outputs are separate Docker mounts; none is part of the image.
Host-side dependencies are Python's standard library and the Docker CLI only.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import tomllib
import urllib.request
import uuid

REVISION = '2299dab3d31f5e93c10df087af86ffb0b7bfe938'
UNITS = ('t2-F1-aud-on-hold-2016', 't2-F3-bear-flattener-2022',
         't2-F4-covid-nfp-2020', 't2-F2-cnh-stress-2015')
GATES = '''
import contextlib,io,json,sys
from pathlib import Path
from qfbench2_common.smoke import run_smoke
from qfbench2_track_forecasting.scoring import _main,build_verifier
import torch
root=Path('/raw-unit');out=Path('/output')
assert torch.version.cuda is None
assert {p.name for p in out.iterdir()}=={'forecast.parquet','forecast_meta.json','forecast_rationale.md'}
rationale=(out/'forecast_rationale.md').read_text(encoding='utf-8')
d=json.loads(rationale.rsplit('```json'+chr(10),1)[1].split('```',1)[0])
expected=4 if d['unit_id']=='t2-F4-covid-nfp-2020' else 5
assert d['config']['ss_scales']==expected
assert d['config']['ss_target_half_life'] is None
parameters=d['generator_parameters']
for component in parameters.get('components',[parameters]):
 if 'target_time_weighting' in component:
  weighting=component['target_time_weighting']
  assert weighting['method']=='uniform' and weighting['half_life'] is None
  assert weighting['reference_energy_weighted'] is False
assert d['config']['training_window'] is None
assert d['config']['ss_device']=='cpu' and d['config']['ss_iterations']==200
assert d['config']['context_length']==(20 if d['unit_id']=='t2-F4-covid-nfp-2020' else 126)
assert d['config']['use_psmc'] is False and d['psmc'] is None
assert d['mixture_alpha']==0 and 'PSMC is disabled' in rationale
assert 'alpha=0' in json.loads((out/'forecast_meta.json').read_text())['rationale']['method']
assert not d['text_used']

import numpy as np
from center_runtime import artifact
c=d['residual_correction'];m,digest=artifact()
assert c['revision']=='sha256:'+digest
assert c['training_cutoff']=='2019-10-23' and c['selection_cutoff']=='2020-07-21'
eligible=d['asof']>c['selection_cutoff']
assert c['width_factor']==(1.5 if eligible else 1.)
assert c['center_applied']==(eligible and 'transfer' not in d and expected==5)
assert c['joint_draw_order_preserved'] and not c['text_used']
q=np.asarray(d['quantiles']);q=q[:,:,None] if q.ndim==2 else q
median=np.asarray(c['original_median']);delta=np.asarray(c['delta'])
before=np.asarray(c['original_quantiles'])
np.testing.assert_allclose(q,median+delta+c['width_factor']*(before-median),rtol=1e-10,atol=1e-12)
assert 'Residual correction' in rationale and 'HistGradientBoosting' in rationale

a,b,e=io.StringIO(),io.StringIO(),io.StringIO()
with contextlib.redirect_stdout(a),contextlib.redirect_stderr(e):
 smoke=run_smoke(root,out,build_verifier)
with contextlib.redirect_stdout(b),contextlib.redirect_stderr(e):
 try: code=_main(['score','--card',str(root/'card.toml'),'--forecast',str(out/'forecast.parquet')])
 except SystemExit as exc: code=exc.code
score=json.loads(b.getvalue())
assert smoke.admissible and code in (None,0) and score['admissible'] is True,(a.getvalue(),b.getvalue(),e.getvalue())
print(json.dumps({'admissible':True,'unit_id':d['unit_id'],'actual_J':expected,
 'context':d['config']['context_length'],'scale_policy':d['scale_policy'],
 'training_rows':d['training_rows'],'cpu_workers':d['config']['ss_workers'],
 'use_psmc':d['config']['use_psmc'],'mixture_alpha':d['mixture_alpha'],
 'center_applied':c['center_applied'],'width_factor':c['width_factor'],'model_revision':c['revision'],
 'target_half_life':d['config']['ss_target_half_life'],'target_weighting':'uniform',
 'official_gates':score,'torch':torch.__version__}))
'''


def command(args, timeout=1800):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {result.stderr[-5000:]} {result.stdout[-5000:]}')
    return result.stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    root=Path(os.environ.get('RUNNER_TEMP', str(Path.cwd()/'tmp')))/('track2-ci-'+uuid.uuid4().hex)
    root.mkdir(parents=True)
    url=f'https://codeload.github.com/Agenthon-2026/track2-forecasting-public/tar.gz/{REVISION}'
    with urllib.request.urlopen(url,timeout=120) as response:
        data=response.read(100*1024*1024)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        for member in archive.getmembers():
            parts=Path(member.name).parts
            if len(parts)<4 or parts[1]!='units' or parts[2] not in UNITS:
                continue
            relative=Path(*parts[3:])
            if not member.isfile() or '..' in relative.parts:
                continue
            with archive.extractfile(member) as source:
                content=source.read()
            # Keep the committed tree intact for official checksum verification.
            raw_path=root/'raw-units'/parts[2]/relative
            assert raw_path.resolve().is_relative_to(root.resolve())
            raw_path.parent.mkdir(parents=True,exist_ok=True)
            raw_path.write_bytes(content)
            if len(relative.parts)==1 and relative.suffix=='.parquet':
                relative=Path('panels')/relative
            elif not (str(relative) in ('card.toml','forecast_spec.json') or relative.parts[0]=='text'):
                continue
            path=root/'inputs'/parts[2]/relative
            assert path.resolve().is_relative_to(root.resolve())
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(content)
    host_cpus=min(16,int(command(['docker','info','--format','{{.NCPU}}']).strip()))
    report=dict(image=args.image,source_revision=REVISION,accuracy_scored=False,
                cpu_limit=host_cpus,results=[])
    # Bound tmpfs, privileges and process count as on the Development worker.
    limits=['docker','run','--rm','--network','none','--read-only','--user','65534:65534',
        '--cap-drop','ALL','--security-opt','no-new-privileges',
        '--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=64m','--pids-limit','256',
        '--ulimit','nofile=1024:1024','--ulimit','nproc=256:256',
        '--ulimit','fsize=67108864:67108864','--cpus',str(host_cpus),'--memory','128g','--memory-swap','128g']
    for unit in UNITS:
        input_path=root/'inputs'/unit
        card=tomllib.loads((input_path/'card.toml').read_text(encoding='utf-8'))
        output_path=root/'outputs'/unit;output_path.mkdir(parents=True);output_path.chmod(0o777)
        mounts=['--mount',f'type=bind,source={input_path},target=/input,readonly',
                '--mount',f'type=bind,source={root/"raw-units"/unit},target=/raw-unit,readonly',
                '--mount',f'type=bind,source={output_path},target=/output']
        started=time.perf_counter()
        command(limits+mounts+['-e','QFBENCH_SEED=0','-e','QFBENCH_NETWORK=none',args.image,
            'forecast','--panels','/input/panels','--text','/input/text','--asof',
            card['provenance']['data_cutoff'],'--out','/output/forecast.parquet'])
        elapsed=time.perf_counter()-started
        gate=json.loads(command(limits+mounts+['--entrypoint','python',args.image,'-c',GATES]))
        report['results'].append(dict(unit_id=unit,seconds=elapsed,**{k:v for k,v in gate.items() if k!='unit_id'}))
        print(f'PASS {unit}: {elapsed:.1f}s, J={gate["actual_J"]}',flush=True)
    # Compare a genuine rerun of one forecast with the same seed.
    unit=UNITS[0];input_path=root/'inputs'/unit;previous=root/'outputs'/unit
    output_path=root/'repeat';output_path.mkdir();output_path.chmod(0o777)
    card=tomllib.loads((input_path/'card.toml').read_text(encoding='utf-8'))
    command(limits+['--mount',f'type=bind,source={input_path},target=/input,readonly',
        '--mount',f'type=bind,source={output_path},target=/output','-e','QFBENCH_SEED=0',
        '-e','QFBENCH_NETWORK=none',args.image,'forecast','--panels','/input/panels',
        '--text','/input/text','--asof',card['provenance']['data_cutoff'],'--out','/output/forecast.parquet'])
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(previous/'forecast.parquet')==sha(output_path/'forecast.parquet')
    report['same_seed_forecast_identical']=True
    report['checks_passed']=len(report['results'])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+chr(10),encoding='utf-8')
    print('IMAGE_VALIDATION_JSON='+json.dumps(report,separators=(',',':')),flush=True)


if __name__=='__main__':
    main()
