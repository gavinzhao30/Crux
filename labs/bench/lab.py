"""Bench lab (laptop side): four rival screen-to-camera response laws and a client for the node.

Designs are grey levels 0..255; the observation is the camera mean on the same 0..255 scale.

Hypotheses (``x`` is the grey level, ``y`` the measured value) and their priors. Every prior has
a free offset ``b ~ Normal(0, 8)``, and the scale of each curve is drawn so that it ends up near
255 at x = 255 (times a factor in 0.6..1.4 for the shape families, or in 0.4..1.6 for the line),
so each family can plausibly explain a monotone camera curve before seeing data:

* linear      y = a*x + b                          a ~ U(0.4, 1.6)
* power       y = 255*a*(x/255)**g + b             a ~ N(1, 0.15) clipped to [0.5, 1.5],
                                                  g ~ U(0.3, 2.0)
* log         y = a*log1p(k*x) + b                 k log-uniform in [0.003, 0.3],
                                                  a = s*255/log1p(255k), s ~ U(0.6, 1.4)
* saturating  y = a*(1 - exp(-k*x)) + b            k log-uniform in [0.001, 0.03],
                                                  a = s*255/(1 - exp(-255k)), s ~ U(0.6, 1.4)

The power gain ``a`` is deliberately tight around 1: a camera whose exposure is locked has
gain near 1, so a tint over the lens (gain 0.5) is not something power can absorb and shows up
as surprise. Observation noise ``sigma`` defaults to 2.0.
"""

import asyncio
import json
import urllib.request

import numpy as np

from core.eig_exec import ModelFamily

DEFAULT_SIGMA = 2.0
OFFSET_SD = 8.0


def _x(designs: np.ndarray) -> np.ndarray:
    return designs[:, 0][None, :]


def _log_uniform(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def linear_family() -> ModelFamily:
    def predict(designs, params):
        return params[:, :1] * _x(designs) + params[:, 1:2]

    def sample_prior(rng, n):
        return np.column_stack(
            [rng.uniform(0.4, 1.6, n), rng.normal(0.0, OFFSET_SD, n)]
        )

    return ModelFamily("linear", predict, sample_prior)


def power_family() -> ModelFamily:
    def predict(designs, params):
        base = np.clip(_x(designs) / 255.0, 0.0, None)
        return 255.0 * params[:, :1] * base ** params[:, 1:2] + params[:, 2:3]

    def sample_prior(rng, n):
        a = np.clip(rng.normal(1.0, 0.15, n), 0.5, 1.5)
        return np.column_stack(
            [a, rng.uniform(0.3, 2.0, n), rng.normal(0.0, OFFSET_SD, n)]
        )

    return ModelFamily("power", predict, sample_prior)


def log_family() -> ModelFamily:
    def predict(designs, params):
        return (
            params[:, :1] * np.log1p(np.maximum(params[:, 1:2], 1e-9) * _x(designs))
            + params[:, 2:3]
        )

    def sample_prior(rng, n):
        k = _log_uniform(rng, 0.003, 0.3, n)
        a = rng.uniform(0.6, 1.4, n) * 255.0 / np.log1p(255.0 * k)
        return np.column_stack([a, k, rng.normal(0.0, OFFSET_SD, n)])

    return ModelFamily("log", predict, sample_prior)


def saturating_family() -> ModelFamily:
    def predict(designs, params):
        return (
            params[:, :1] * (1.0 - np.exp(-params[:, 1:2] * _x(designs)))
            + params[:, 2:3]
        )

    def sample_prior(rng, n):
        k = _log_uniform(rng, 0.001, 0.03, n)
        a = rng.uniform(0.6, 1.4, n) * 255.0 / (1.0 - np.exp(-255.0 * k))
        return np.column_stack([a, k, rng.normal(0.0, OFFSET_SD, n)])

    return ModelFamily("saturating", predict, sample_prior)


def bench_families() -> list:
    return [linear_family(), power_family(), log_family(), saturating_family()]


def exposure_families() -> list:
    """Same rival curve shapes with broader brightness/offset priors for a lit scene.

    Input is a normalized exposure setting, not display brightness. The scene's gain
    and black offset are unknown; a prior near full-scale screen output is inappropriate.
    """
    result = []
    for family in bench_families():

        def sample_prior(rng, n, base=family.sample_prior, name=family.name):
            params = base(rng, n)
            params[:, -1] = rng.normal(0.0, 25.0, n)
            if name in ("linear", "power"):
                params[:, 0] = rng.uniform(0.15, 1.5, n)
            else:
                params[:, 0] *= rng.uniform(0.25, 1.2, n)
            return params

        result.append(ModelFamily(family.name, family.predict, sample_prior))
    return result


class BenchClient:
    """urllib-only client for ``labs/bench/node.py``."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def patch(self, level: float) -> dict:
        return self._call("POST", "/patch", {"level": round(level)})

    def measure(self) -> dict:
        return self._call("GET", "/measure")

    def health(self) -> dict:
        return self._call("GET", "/health")

    def tint(self, factor: float) -> dict:
        return self._call("POST", "/tint", {"factor": float(factor)})

    def seed(self, seed: int) -> dict:
        return self._call("POST", "/seed", {"seed": int(seed)})


class BenchLab:
    """The numeric lab the orchestrator can drive: candidate designs, families, and ``run``."""

    def __init__(
        self, client: BenchClient, levels=None, max_measurements: int = 10
    ) -> None:
        self.client = client
        self.levels = [
            float(v) for v in (levels if levels is not None else range(0, 256, 17))
        ]
        self.max_measurements = max_measurements

    def candidate_designs(self) -> list:
        return list(self.levels)

    def families(self, sigma: float = DEFAULT_SIGMA) -> list:
        del sigma  # the noise level belongs to the belief, not to the families
        return bench_families()

    async def run(self, level: float) -> float:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.client.patch, level)
        res = await loop.run_in_executor(None, self.client.measure)
        return float(res["mean"])
