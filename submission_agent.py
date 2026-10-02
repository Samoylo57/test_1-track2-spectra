"""Offline CPU competition entrypoint: Scattering Spectra + foveal PSMC."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from prediction import ForecastConfig, Prediction, forecast_history, historical_steps, load_task, resolve_horizon_steps, write_json
from qfbench2_track_forecasting.limits import ParseLimits

ARTIFACTS = {'forecast.parquet','forecast_meta.json','forecast_rationale.md'}

def runtime_seed():
    value = os.environ.get('QFBENCH_SEED','0')
    if not value.isascii() or not value.isdigit():
        raise ValueError('QFBENCH_SEED must be a non-negative integer')
    return int(value)

def profile(frequency, seed):
    return ForecastConfig(generator='scattering-spectra', embedding='foveal',
        context_length=20 if frequency=='monthly' else 126, ss_device='cpu', seed=seed,
        ss_iterations=200,ss_strict_tolerance=False)

def select_fit_segment(series, target_type, frequency, config):
    """An isolated transfer anchor is not a return across the missing decade.

    Prefer the latest segment. If too short for Spectra, use the most recent
    sufficiently long earlier segment, with PSMC context from THAT segment.
    No recent path is inferred or filled in; the final supplied level is an anchor.
    """
    dates = series.index
    if frequency=='monthly':
        gaps = np.diff(dates.to_period('M').asi8) != 1
    else:
        days = np.diff(dates.asi8)/pd.Timedelta(days=1).value
        gaps = days > max(5.,10.*float(np.median(days)))
    bounds = [0,*list(np.flatnonzero(gaps)+1),len(series)]
    for start,end in reversed(list(zip(bounds[:-1],bounds[1:]))):
        segment = series.iloc[start:end]
        if len(segment) < max(65,config.context_length+1):
            continue
        count = len(historical_steps(segment.iloc[:,0] if isinstance(segment,pd.DataFrame) else segment,
            target_type,frequency,config.level_transform))
        if count >= max(64,config.context_length):
            return segment, end!=len(series)
    raise ValueError('no supplied uninterrupted segment long enough for Scattering Spectra')

def predict_task(args):
    card,series,source,frequency,asof = load_task(args,allow_joint=True)
    config = profile(frequency,runtime_seed())
    limits = ParseLimits()
    if not max(limits.min_draws,card.get('scoring',{}).get('params',{}).get('n_draws_min',0)) <= config.n_draws <= limits.max_draws:
        raise ValueError('profile draw count conflicts with card')
    steps = resolve_horizon_steps(card,series,source,frequency,asof)
    fit_series,transfer = select_fit_segment(series,card['targets']['target_type'],frequency,config)
    prediction = forecast_history(fit_series,steps,target_type=card['targets']['target_type'],frequency=frequency,config=config)
    diagnostics = dict(prediction.diagnostics)
    if transfer:
        if card['targets']['target_type']!='level':
            raise ValueError('sparse transfer adjustment supports level targets only')
        old = np.asarray(fit_series.iloc[-1]); current = np.asarray(series.iloc[-1])
        shift = current-old
        diagnostics['transfer'] = dict(reason='isolated latest anchor after an input-history gap',
            fit_last_date=str(fit_series.index[-1].date()), anchor_date=str(series.index[-1].date()),
            old_anchor=old.tolist(), new_anchor=current.tolist(), anchor_shift=shift.tolist(),
            limitation='early historical regime and early PSMC context; no recent target path or text adaptation')
        prediction = Prediction(prediction.samples+shift,prediction.unconditional_samples+shift,diagnostics)
        diagnostics['anchor'] = current.tolist()
        diagnostics['quantiles'] = np.quantile(prediction.samples,[.05,.5,.95],axis=0).tolist()
    diagnostics['unit_id'] = card['task']['id']
    diagnostics['asof'] = asof
    diagnostics['horizon_keys'] = card['targets']['horizons']
    diagnostics['text_used'] = False
    # Evidence identifies only input files in this invocation, never a data index.
    root = source.parent.parent if source.parent.name=='panels' else source.parent
    paths = sorted((root/'panels').glob('*.parquet')) if (root/'panels').is_dir() else sorted(root.glob('*.parquet'))
    diagnostics['input_panels'] = [{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    return card,prediction,diagnostics

def write_artifacts(card,prediction,diagnostics,out):
    if out.name!='forecast.parquet':
        raise ValueError('--out must name forecast.parquet')
    if out.parent.exists() and any(p.name not in ARTIFACTS or not p.is_file() or p.is_symlink() for p in out.parent.iterdir()):
        raise ValueError('output directory may contain only the three regular forecast artifacts')
    out.parent.mkdir(parents=True,exist_ok=True)
    assets = card['targets']['asset_ids']; horizons = card['targets']['horizons']
    n_draws = len(prediction.samples)
    cube = prediction.samples.transpose(0,2,1) if prediction.samples.ndim==3 else prediction.samples[:,None,:]
    if cube.shape!=(n_draws,len(assets),len(horizons)) or not np.isfinite(cube).all():
        raise ValueError('invalid joint draw grid')
    pd.DataFrame(dict(draw=np.repeat(np.arange(n_draws,dtype=np.int64),len(assets)*len(horizons)),
        asset=np.tile(np.repeat(assets,len(horizons)),n_draws),
        horizon=np.tile(np.asarray(horizons,dtype=np.int64),n_draws*len(assets)),value=cube.reshape(-1))).to_parquet(out,index=False)
    write_json(out.parent/'forecast_meta.json',dict(unit_id=card['task']['id'],asof=diagnostics['asof'],
        representation='samples',asset_ids=assets,horizons=horizons,n_draws=n_draws,target=card['targets']['target_type'],
        rationale=dict(file='forecast_rationale.md',method='CPU Scattering Spectra + foveal PSMC; no text or LLM')))
    text = f"# Forecast derivation: {card['task']['id']}\n\n"
    text += f"1. **Anchor.** {assets}, supplied observation at {diagnostics.get('transfer',{}).get('anchor_date',diagnostics['training_last_date'])}; value(s) {diagnostics['anchor']}. Cutoff: {diagnostics['asof']}. Log-return targets start at zero; panels contain decimal simple returns transformed with log(1+r).\n\n"
    text += f"2. **Numerical adjustments.** Fit {diagnostics['training_rows']} increments from {diagnostics['training_first_date']} through {diagnostics['training_last_date']}. Fixed Battle-Lemarie wavelets and Phi1-Phi4 constrain synthetic paths; trajectory values, not neural weights, are optimized. Multiple targets use a prefix PCA and independent Spectra latent continuations, then joint PSMC; this is our explicit multivariate extension.\n\n"
    text += "3. **Conditioning.** Foveal cumulative changes compare observed and generated pasts; the 512 nearest pasts receive Gaussian weights with a past-only bandwidth targeting ESS 256. Whole joint continuations are resampled. All horizons and assets within a draw share that continuation. The synthesis seed is fixed at 12345 (plus latent component index); runtime QFBENCH_SEED controls candidate and resampling draws.\n\n"
    if 'transfer' in diagnostics:
        text += f"4. **Transfer limitation.** {json.dumps(diagnostics['transfer'],ensure_ascii=False)}. No differences span the missing interval.\n\n"
    text += "5. **Scale and shape.** Empirical synthetic increments retain fitted means and scales; their cumulative sums determine endpoint draws. Synthesis uses at most 200 iterations per latent component, aiming at relative error 0.006; finite approximations are retained if this target is not attained. Actual matching errors and convergence flags are recorded below. A non-converged bank is approximate, not a claim of exact spectrum matching. Monthly observation periods use the official mapping while output retains card horizon keys. No hand-coded scenario shifts, text claims, House calls, pretrained weights or other-unit histories are used. Accuracy against sealed future outcomes has not been measured locally.\n\n"
    text += "**Adjustment ledger:** level = supplied last anchor + sum of simulated increments; cumulative log return = sum of simulated log increments. Text adjustment = 0. Transfer cases relocate the old level anchor to the current supplied anchor without estimating the missing path. Values, dates, means, scales, quantiles, PCA loadings and conditioning diagnostics follow.\n\n```json\n"
    text += json.dumps(diagnostics,ensure_ascii=False,indent=2,allow_nan=False)+'\n```\n'
    (out.parent/'forecast_rationale.md').write_text(text,encoding='utf-8')
    if sum(p.stat().st_size for p in out.parent.iterdir())>64*1024**2 or (out.parent/'forecast_rationale.md').stat().st_size>ParseLimits().max_rationale_bytes:
        raise ValueError('artifacts exceed competition size limits')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb',choices=['forecast'])
    parser.add_argument('--panels',type=Path)
    parser.add_argument('--text',type=Path)
    parser.add_argument('--card',type=Path)
    parser.add_argument('--asof')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--unit-id',help='local development convenience; image has no unit index')
    args = parser.parse_args(argv)
    try:
        card,prediction,diagnostics = predict_task(args)
        write_artifacts(card,prediction,diagnostics,args.out.resolve())
        print(json.dumps(dict(unit_id=card['task']['id'],n_draws=len(prediction.samples),
            assets=len(card['targets']['asset_ids']),seed=runtime_seed(),device='cpu',
            transfer='transfer' in diagnostics,psmc_ess=diagnostics['psmc']['effective_sample_size'])))
    except (ValueError,KeyError,OSError,RuntimeError) as exc:
        parser.exit(2,f'forecast failed: {exc}\n')
    return 0

if __name__=='__main__':
    sys.exit(main())
