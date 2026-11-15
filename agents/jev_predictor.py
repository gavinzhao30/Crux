"""Batch conditional diagnostic predictions through the TypeSafe Choice API."""

import hashlib
import json
import math
import tempfile
from pathlib import Path

import httpx

from agents.predictor import diagnostic_spec
from core.schemas import Experiment, Hypothesis

API_URL = "https://api.typesafe.ai/v1"


def probability_row(raw: dict, bins: list[str]) -> dict[str, float]:
    if set(raw) != set(bins):
        raise ValueError("Prediction must cover exactly the declared bins")
    values = {key: float(raw[key]) for key in bins}
    total = sum(values.values())
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values.values()) or total <= 0:
        raise ValueError("Bin probabilities must have finite positive total mass")
    return {key: value / total for key, value in values.items()}


async def discover_model(client: httpx.AsyncClient) -> str:
    response = await client.get(f"{API_URL}/models")
    response.raise_for_status()
    available = [
        model for model in response.json()["models"] if model["name"].startswith("jev-")
    ]
    if not available:
        raise ValueError("The TypeSafe account returned no Jev models")
    if any(model["name"] == "jev-latest" for model in available):
        return "jev-latest"
    return max(available, key=lambda m: (m.get("release_date") or "", m["name"]))[
        "name"
    ]


class JevPredictor:
    def __init__(self, client: httpx.AsyncClient, model: str, cache_dir: Path):
        self.client = client
        self.model = model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def predict(
        self, exp: Experiment, hypotheses: list[Hypothesis], context: str
    ) -> dict[str, dict[str, float]]:
        keys = {f"option_{index}": h for index, h in enumerate(hypotheses)}
        criteria = {b.id: f"{b.description}. Rule: {b.rule}" for b in exp.bins}
        payload = {
            "model": self.model,
            "state": {"context": context, "diagnostic": diagnostic_spec(exp)},
            "questions": {
                key: {
                    "type": "choice",
                    "instructions": (
                        f"Assume this hypothesis is true: {h.statement} "
                        "Which predeclared bin will the diagnostic produce?"
                    ),
                    "criteria": criteria,
                }
                for key, h in keys.items()
            },
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if path.is_file():
            response = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = await self.client.post(f"{API_URL}/systemone", json=payload)
            result.raise_for_status()
            response = result.json()
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.cache_dir, delete=False
            ) as handle:
                json.dump(response, handle)
            Path(handle.name).replace(path)
        return {
            h.id: probability_row(
                response["answers"][key]["probabilities"], list(criteria)
            )
            for key, h in keys.items()
        }
