"""Exact frozen past-only feature definition for the residual centre model."""
import numpy as np

FEATURE_NAMES = ('spectra_drift','log_horizon','last_step','momentum_5','momentum_20','momentum_60',
                 'log_vol5_vol252','log_vol20_vol252','log_vol60_vol252',
                 'log_spectra_vol252','spectra_asymmetry','is_log_return','is_rate','rate_level')
MIN_CENTER = 80
MIN_WIDTH = 100


def features(values, draws, horizons, assets, target_type):
    """Only past values and predicted draws; return [cell,feature] in H,A order."""
    v,x=np.asarray(values,float),np.asarray(draws,float)
    if (v.ndim!=2 or x.ndim!=3 or len(v)<257 or v.shape[1]!=x.shape[2]
            or x.shape[1]!=len(horizons) or len(assets)!=v.shape[1]
            or not np.isfinite(v).all() or not np.isfinite(x).all()):
        raise ValueError('finite aligned history, joint draws and horizons required')
    if target_type=='level': increments=np.diff(v,axis=0); anchor=v[-1]
    elif target_type=='log_return':
        if np.any(v<=-1): raise ValueError('invalid simple returns')
        increments=np.log1p(v); anchor=np.zeros(v.shape[1])
    else: raise ValueError('unsupported target')
    median=np.median(x,axis=0); sd=np.std(x,axis=0,ddof=1)
    step_sd=np.std(increments[-252:],axis=0,ddof=1)
    denominator=np.maximum(step_sd,1e-12)
    forecast_sd=np.maximum(sd,1e-12)
    h=np.asarray(horizons,float)[:,None]
    rows=[(median-anchor)/forecast_sd,np.broadcast_to(np.log1p(h),median.shape)]
    for k in (1,5,20,60):
        change=increments[-k:].sum(0)/(denominator*np.sqrt(k))
        rows.append(np.broadcast_to(change,median.shape))
    for k in (5,20,60):
        vol=np.std(increments[-k:],axis=0,ddof=1)
        rows.append(np.broadcast_to(np.log(np.maximum(vol,1e-12)/denominator),median.shape))
    rows.append(np.log(forecast_sd/(denominator*np.sqrt(h))))
    q05,q95=np.quantile(x,[.05,.95],axis=0)
    rows.append((q95+q05-2*median)/forecast_sd)
    rows.append(np.full(median.shape,float(target_type=='log_return')))
    rate=np.array([a.startswith('UST_') for a in assets],float)
    rows.append(np.broadcast_to(rate,median.shape))
    rows.append(np.broadcast_to(rate*anchor,median.shape))
    result=np.stack(rows,axis=-1).reshape(-1,len(FEATURE_NAMES))
    # Fixed numeric protection for near-constant histories, not learned clipping.
    return np.clip(result,-20.,20.),median,sd

