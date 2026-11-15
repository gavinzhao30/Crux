"""Record numerical-reference ranking sensitivity on development seeds, without model calls."""
import json
from pathlib import Path

import numpy as np

from core.eig_exec import ModelFamily, ParticleBelief
from eval.boxing_study import GRID
from labs.boxing.location_finding import LocationFinding, predict_sources, source_prior

rows = []
family = ModelFamily("sources", predict_sources, source_prior, log_prior=lambda p: -.5*(p*p).sum(axis=1))
for seed in [3000, 3001, 3002]:
    for observed in [0, 5]:
        arrays = []
        diagnostics = []
        for particles, simulations in [(512, 64), (2048, 256)]:
            belief = ParticleBelief([family], .5, seed=seed, n_particles=particles,
                                    n_sim=simulations, null_weight=0, move_steps=4)
            env = LocationFinding(seed)
            for point in GRID[np.linspace(0, 48, 5, dtype=int)][:observed]:
                belief.update(point, env.observe(point))
            gains = belief.expected_information_gain(GRID, goal="total")
            arrays.append(gains)
            diagnostics.append({"particles": particles, "simulations": simulations,
                                "top_design": GRID[int(gains.argmax())].tolist(),
                                "max_eig_bits": float(gains.max()), "moves": belief.move_diagnostics})
        ranks = [np.argsort(np.argsort(a)) for a in arrays]
        correlation = float(np.corrcoef(*ranks)[0, 1])
        rows.append({"seed": seed, "observations": observed,
                     "rank_correlation": correlation,
                     "same_top_design": bool(arrays[0].argmax() == arrays[1].argmax()),
                     "settings": diagnostics})
output = Path("eval/results/boxing/repair-numerics-20261115.json")
output.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
