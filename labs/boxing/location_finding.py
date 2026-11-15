"""Location Finding, ported from kanishkg/boxing-gym (MIT), Signal/DirectGoal.

Upstream commit: 5df3d32cfa7e4de99b43bb4d5328e888cc0cb7b9.
Upstream: src/boxing_gym/envs/location_finding.py. Observations are rounded;
evaluation uses noiseless intensities. Random inputs use Uniform(-2, 2).
"""

import numpy as np


def intensity(points, sources):
    points, sources = np.asarray(points), np.asarray(sources)
    return 0.1 + (1 / (1e-4 + ((points[..., None, :] - sources) ** 2).sum(axis=-1))).sum(axis=-1)


class LocationFinding:
    sigma = 0.5

    def __init__(self, seed):
        streams = np.random.SeedSequence(seed).spawn(3)
        self._sources = np.random.default_rng(streams[0]).normal(size=(3, 2))
        self._noise = np.random.default_rng(streams[1])
        self._test = np.random.default_rng(streams[2]).normal(size=(10, 2))
        self.history = []

    def observe(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Measurement must be a finite 2-D point")
        value = round(float(intensity(point, self._sources)) + self._noise.normal(0, self.sigma), 2)
        self.history.append((point.tolist(), value))
        return value

    def score(self, predictions):
        predictions = np.asarray(predictions, dtype=float)
        if predictions.shape != (10,) or not np.isfinite(predictions).all():
            raise ValueError("Expected ten finite predictions")
        return float(np.mean((predictions - intensity(self._test, self._sources)) ** 2))

    def evaluation_points(self):
        return self._test.copy()


def predict_sources(designs, params):
    return np.stack([intensity(designs, p.reshape(3, 2)) for p in params])


def source_prior(rng, n):
    return rng.normal(size=(n, 6))
