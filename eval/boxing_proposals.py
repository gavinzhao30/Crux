"""Bounded, auditable proposal repair shared by study and Modal harness."""
import asyncio

from core.model_validation import validate_family


async def generate_proposals(llm, seed):
    from eval.boxing_study import Families, FamilyProposal, ask

    attempts, accepted = [], []
    try:
        batch = await ask(llm, Families, {"task": "Propose two distinct executable numpy models. "
            "Define deterministic predict(designs, params) returning exactly (n_particles,n_designs), "
            "and sample_prior(rng,n) returning exactly (n,p), p>0. Use only numpy; no I/O. "
            "Use the supplied rng for prior sampling. Predictions are noiseless means. "
            "Do not reshape designs using the number of particles. Sum contributions from sources."})
        for index, original in enumerate(batch.families):
            proposal = original
            for repair in range(3):
                error = None
                try:
                    await asyncio.to_thread(validate_family, proposal.code, seed)
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"[:4000]
                attempts.append({"family": index, "repair": repair, "proposal": proposal.model_dump(), "error": error})
                if error is None:
                    accepted.append(proposal)
                    break
                if repair < 2:
                    proposal = await ask(llm, FamilyProposal, {"task": "Repair this model to satisfy the exact array contract. "
                        "Keep its scientific hypothesis. predict must be deterministic; sample_prior must use supplied rng.",
                        "proposal": proposal.model_dump(), "validation_error": error})
        if len(accepted) != len(batch.families):
            raise ValueError("One or more models failed after two repairs")
        return {"families": [p.model_dump() for p in accepted], "attempts": attempts, "validation_error": None}
    except Exception as exc:  # noqa: BLE001
        return {"families": [], "attempts": attempts, "validation_error": f"{type(exc).__name__}: {exc}"[:4000]}
