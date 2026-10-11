"""Joint extension: fixed prefix PCA, Spectra latent paths, joint PSMC.

This is an explicit extension of the univariate pilot, not the paper's
multivariate scattering operator. Every loading is estimated on this unit only.
"""
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import numpy as np
from scattering_generator import ScatteringSpectraGenerator, load_backend
from scattering_cpu import component_workers, single_threaded_backend

@dataclass
class JointScatteringGenerator:
    options: dict = field(default_factory=dict)
    workers: int = 1

    def __post_init__(self):
        if not isinstance(self.workers, int) or not 1 <= self.workers <= 16:
            raise ValueError('component workers must be an integer in [1,16]')

    def fit(self, increments):
        values = np.asarray(increments, dtype=float)
        if values.ndim != 2 or len(values) < 64 or not np.isfinite(values).all():
            raise ValueError('joint Spectra needs >=64 aligned finite increments')
        self.mean = values.mean(0)
        scale = values.std(0, ddof=1)
        self.scale = np.where(scale > 1e-12, scale, 1.)
        centered = (values-self.mean)/self.scale
        covariance = centered.T @ centered / (len(values)-1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:,order]
        # Fix eigenvector signs for reproducibility, including eigensolver sign flips.
        for i in range(len(eigenvalues)):
            pivot = np.argmax(np.abs(eigenvectors[:,i]))
            if eigenvectors[pivot,i] < 0:
                eigenvectors[:,i] *= -1
        active = eigenvalues > max(float(eigenvalues.max()),1.) * 1e-10
        self.loadings = eigenvectors[:,active] * np.sqrt(eigenvalues[active])
        latent = centered @ eigenvectors[:,active] / np.sqrt(eigenvalues[active])
        self.models = []
        for i in range(latent.shape[1]):
            options = dict(self.options)
            options['seed'] = int(options.get('seed',12345)) + i
            self.models.append(ScatteringSpectraGenerator(**options).fit(latent[:,i]))
        self.eigenvalues = eigenvalues
        self._execution = {}
        return self

    def _prepare_banks(self):
        missing = [model for model in self.models if model._bank is None
                   and model._scale > max(1., abs(model._mean)) * 1e-12]
        if not missing:
            return
        workers = (component_workers(self.workers, len(missing))
                   if self.options.get('device', 'cpu') == 'cpu' else 1)
        started = time.perf_counter()
        if workers == 1:
            for model in missing:
                model._synthesize()
        else:
            torch, Model = load_backend()
            cancelled = threading.Event()
            # Torch/SciPy kernels release the GIL. Threads share imported code
            # and filters without IPC, extra interpreters or temporary files.
            with single_threaded_backend(torch):
                pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='spectra')
                futures = []
                try:
                    futures = [pool.submit(model._synthesize_limited, torch, Model, cancelled)
                               for model in missing]
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    cancelled.set()
                    for future in futures:
                        future.cancel()
                    raise
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
        self._execution = dict(component_workers_requested=self.workers,
            component_workers_used=workers, bank_preparation_seconds=time.perf_counter()-started)

    def sample(self, n_paths, n_steps, rng):
        if not hasattr(self,'models') or n_paths<1 or n_steps<1:
            raise ValueError('fit joint Spectra first and supply positive sizes')
        if self.models and n_steps > self.models[0].max_path_length:
            raise ValueError('requested steps exceed ss-max-path-length')
        self._prepare_banks()
        paths = np.broadcast_to(self.mean,(n_paths,n_steps,len(self.mean))).copy()
        for i,model in enumerate(self.models):
            paths += model.sample(n_paths,n_steps,rng)[:,:,None] * (self.loadings[:,i]*self.scale)
        return paths

    def diagnostics(self):
        return dict(extension='prefix PCA + independent univariate Spectra latent continuations',
            fitted_mean=self.mean.tolist(), asset_scales=self.scale.tolist(),
            eigenvalues=self.eigenvalues.tolist(), loadings=self.loadings.tolist(),
            components=[m.diagnostics() for m in self.models], **self._execution)
