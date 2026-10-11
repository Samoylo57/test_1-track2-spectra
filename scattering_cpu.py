"""CPU execution adapters; the pinned Scattering Spectra sources stay unchanged.

No compilation, disk-backed arrays, extra data or learned weights are involved.
The adapters remove identity gathers and scope native thread limits around a
whole synthesis group, rather than changing global limits inside its workers.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import multiprocessing
import os
from pathlib import Path
import threading
from types import MethodType

_BACKEND_LOCK = threading.RLock()
_CPU_ALLOCATION = ContextVar('spectra_cpu_allocation', default=None)


def cpu_budget():
    """Respect process affinity and Linux container quotas, capped at 16 CPUs."""
    limits = [16, getattr(os, 'process_cpu_count', os.cpu_count)() or 1]
    if hasattr(os, 'sched_getaffinity'):
        limits.append(len(os.sched_getaffinity(0)))
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            limits.append(max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError):
        try:
            root = Path('/sys/fs/cgroup/cpu')
            quota = int((root / 'cpu.cfs_quota_us').read_text())
            period = int((root / 'cpu.cfs_period_us').read_text())
            if quota > 0:
                limits.append(max(1, math.floor(quota / period)))
        except (OSError, ValueError):
            pass
    return max(1, min(limits))


def component_workers(requested, components):
    # Evaluation already distributes cards to processes. Do not multiply that
    # outer pool by a second pool of component workers.
    allocation = _CPU_ALLOCATION.get()
    if multiprocessing.parent_process() is not None and allocation is None:
        return 1
    return max(1, min(requested, components, cpu_budget(), allocation or 16))


@contextmanager
def allocated_cpu_budget(slots):
    """An outer scheduler can grant bounded slots to one evaluation card.

    This enables component threads inside its process without multiplying the
    outer process count by an unbudgeted inner pool. The grant is local to the
    invocation and is reset even when a card fails.
    """
    if not isinstance(slots, int) or not 1 <= slots <= 16:
        raise ValueError('allocated CPU slots must be an integer in [1,16]')
    token = _CPU_ALLOCATION.set(slots)
    try:
        yield
    finally:
        _CPU_ALLOCATION.reset(token)


@contextmanager
def single_threaded_backend(torch):
    from threadpoolctl import threadpool_limits
    with _BACKEND_LOCK:
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with threadpool_limits(limits=1):
                yield
        finally:
            torch.set_num_threads(previous)


def _channelwise_correlation(self, sxl, sxr=None, multivariate=True):
    if sxr is None:
        sxr = sxl
    if multivariate or sxl.shape[1] != sxr.shape[1]:
        return type(self).forward(self, sxl, sxr, multivariate)
    # Channel selection in the upstream diagonal operator is arange(N): an
    # identity gather that copies both large tensors and adds two backward
    # scatter operations. Keep the original scale gathers and reduction order.
    xl = sxl[:, :, self._cpu_idx_l, 0, :]
    xr = sxr[:, :, self._cpu_idx_r, 0, :]
    return self.ave(xl * xr.conj())


def install_cpu_adapters(model, torch):
    """Replace diagonal channel identity gathers only, without changing Phi."""
    for name in ('module_corr_w', 'module_corr_wmw', 'module_corr_mw'):
        module = getattr(model, name, None)
        if module is None or hasattr(module, '_cpu_idx_l'):
            continue
        module.register_buffer('_cpu_idx_l', torch.tensor(module.idx_l.copy(), dtype=torch.long))
        module.register_buffer('_cpu_idx_r', torch.tensor(module.idx_r.copy(), dtype=torch.long))
        module.forward = MethodType(_channelwise_correlation, module)
    return model


def coefficients(model, x, fastpath):
    # The synthesis driver sends at most one batch and uses a shared prefix
    # normalization. Avoid copying that whole batch through x[arange(B)].
    sigma = model.norm_layer.sigma
    if fastpath and x.shape[0] <= model.nchunks and (sigma is None or sigma.shape[0] == 1):
        return model.forward_batch(x).y
    return model(x).y
