"""Joint extension: fixed prefix PCA, Spectra latent paths, joint PSMC.

This is an explicit extension of the univariate pilot, not the paper's
multivariate scattering operator. Every loading is estimated on this unit only.
"""
from dataclasses import dataclass, field
import numpy as np
from scattering_generator import ScatteringSpectraGenerator

@dataclass
class JointScatteringGenerator:
    options: dict = field(default_factory=dict)

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
        return self

    def sample(self, n_paths, n_steps, rng):
        if not hasattr(self,'models') or n_paths<1 or n_steps<1:
            raise ValueError('fit joint Spectra first and supply positive sizes')
        paths = np.broadcast_to(self.mean,(n_paths,n_steps,len(self.mean))).copy()
        for i,model in enumerate(self.models):
            paths += model.sample(n_paths,n_steps,rng)[:,:,None] * (self.loadings[:,i]*self.scale)
        return paths

    def diagnostics(self):
        return dict(extension='prefix PCA + independent univariate Spectra latent continuations',
            fitted_mean=self.mean.tolist(), asset_scales=self.scale.tolist(),
            eigenvalues=self.eigenvalues.tolist(), loadings=self.loadings.tolist(),
            components=[m.diagnostics() for m in self.models])
