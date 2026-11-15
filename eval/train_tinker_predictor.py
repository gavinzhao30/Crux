"""LoRA SFT of the outcome predictor on Tinker (DU-29).

Trains ``Qwen/Qwen3-8B`` (LoRA) to emit the observed bin of a library diagnostic, given the
prompt from ``agents.tinker_predictor.build_prompt`` (diagnostic goal, bins, the true option
and the compact data profile). Cross-entropy weights are 1 on the completion tokens
(``bin_completion``: the bin id and the end token) and 0 on the prompt.

* Spend cap. Tokens are counted with the model's tokenizer before anything runs; the
  estimated cost (training tokens for every planned epoch plus the validation passes, all at
  the training price) is printed, and the run refuses to start when the estimate plus what
  ``runs/predictor_data/spend.json`` already records exceeds ``--max-usd`` (default 15).
* Probe. Step 0 trains on one example. A billing or credit error stops the run cleanly with
  exit code 3 and ``train_status.json`` saying why.
* Early stopping. Validation loss (mean negative log-probability of the true bin's completion,
  per example) is measured every ``--eval-every`` steps on a fixed validation subset; each
  improvement saves sampler weights, and ``--patience`` evaluations without improvement stop.
* Output. ``runs/predictor_data/model.json`` holds the best sampler path (``tinker://...``),
  the hyperparameters and the losses; ``train_log.jsonl`` one line per step and evaluation.
  No secret is ever written; the API key is read by the SDK from ``TINKER_API_KEY``.

    set -a; . /path/to/.env; set +a
    uv run --group tinker python -m eval.train_tinker_predictor --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from eval.predictor_data.dataset import (
    DATA_DIR,
    add_to_ledger,
    load_split,
    read_ledger,
    record_example,
)

BASE_MODEL = "Qwen/Qwen3-8B"
# $ per million tokens (Tinker list prices for Qwen3-8B).
TRAIN_PRICE = {"Qwen/Qwen3-8B": 0.44}
PREFILL_PRICE = {"Qwen/Qwen3-8B": 0.195}


# --- data -----------------------------------------------------------------------------


def sft_arrays(
    prompt_tokens: list[int], completion_tokens: list[int]
) -> dict[str, list]:
    """Next-token SFT arrays: input is all tokens but the last, targets shifted by one,
    weights 1 only where the target is a completion token."""
    if not completion_tokens:
        raise ValueError("empty completion")
    tokens = list(prompt_tokens) + list(completion_tokens)
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return {"input": tokens[:-1], "targets": tokens[1:], "weights": weights[1:]}


def to_datum(arrays: dict[str, list]):
    from tinker import types

    return types.Datum(
        model_input=types.ModelInput.from_ints(arrays["input"]),
        loss_fn_inputs={
            "target_tokens": arrays["targets"],
            "weights": arrays["weights"],
        },
    )


def encode_examples(records: list[dict], tokenizer) -> list[dict]:
    out = []
    for record in records:
        prompt, completion = record_example(record)
        arrays = sft_arrays(
            tokenizer.encode(prompt, add_special_tokens=False),
            tokenizer.encode(completion, add_special_tokens=False),
        )
        arrays["case_id"] = record["case_id"]
        out.append(arrays)
    return out


def batch_losses(output, batch: list[dict]) -> tuple[float, float]:
    """(mean NLL per completion token, mean NLL per example) from a forward(_backward)."""
    total, tokens, per_example = 0.0, 0.0, []
    for result, arrays in zip(output.loss_fn_outputs, batch, strict=True):
        logprobs = result["logprobs"].tolist()
        nll = -sum(lp * w for lp, w in zip(logprobs, arrays["weights"], strict=True))
        weight = sum(arrays["weights"])
        total += nll
        tokens += weight
        per_example.append(nll)
    return total / max(tokens, 1.0), sum(per_example) / max(len(per_example), 1)


def n_tokens(examples: list[dict]) -> int:
    return sum(len(e["input"]) for e in examples)


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * max(0.05, 1.0 - progress)  # linear decay to 5% of the peak


def estimate_usd(
    train_tokens: int, val_tokens: int, epochs: int, n_evals: int, model: str
):
    price = TRAIN_PRICE.get(model, 1.0)
    usd = (train_tokens * epochs + val_tokens * n_evals) * price / 1e6
    return round(usd, 3)


# --- run ------------------------------------------------------------------------------


def _billing_problem(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    try:
        import tinker

        if isinstance(exc, tinker.BillingError):
            return True
    except ImportError:  # pragma: no cover
        pass
    return any(
        word in text for word in ("billing", "credit", "payment", "402", "quota")
    )


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LoRA SFT of the Crux outcome predictor"
    )
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--val-size", type=int, default=600)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train-cases", type=int, default=None)
    parser.add_argument("--max-usd", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="estimate cost and exit")
    args = parser.parse_args(argv)

    out = Path(args.data)
    status_path, log_path = out / "train_status.json", out / "train_log.jsonl"
    rng = random.Random(args.seed)
    train_records = load_split("train", out)
    if args.max_train_cases:
        keep = sorted({r["case_id"] for r in train_records})[: args.max_train_cases]
        keep_set = set(keep)
        train_records = [r for r in train_records if r["case_id"] in keep_set]
    val_records = load_split("val", out)
    rng.shuffle(val_records)
    val_records = val_records[: args.val_size]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train = encode_examples(train_records, tokenizer)
    val = encode_examples(val_records, tokenizer)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    n_evals = total_steps // args.eval_every + 2
    est = estimate_usd(n_tokens(train), n_tokens(val), args.epochs, n_evals, args.model)
    spent = float(read_ledger(out).get("total_usd", 0.0))
    print(
        f"{len(train)} train examples ({len({r['case_id'] for r in train_records})} cases, "
        f"{n_tokens(train) / 1e6:.2f}M tokens/epoch), {len(val)} val examples; "
        f"{total_steps} steps planned; estimated cost ${est:.2f} "
        f"(+ ${spent:.2f} already in the spend ledger; cap ${args.max_usd:.2f})",
        flush=True,
    )
    if est + spent > args.max_usd:
        print("Refusing to train: the estimate exceeds --max-usd.", flush=True)
        return 2
    if args.dry_run:
        return 0

    import tinker
    from tinker import types

    started = time.monotonic()
    service = tinker.ServiceClient()
    try:
        client = service.create_lora_training_client(
            base_model=args.model, rank=args.rank, seed=args.seed
        ).result()
        remote_tok = client.get_tokenizer()
        sample = val_records[0] if val_records else train_records[0]
        prompt, completion = record_example(sample)
        if remote_tok.encode(
            prompt + completion, add_special_tokens=False
        ) != tokenizer.encode(prompt + completion, add_special_tokens=False):
            raise RuntimeError("local and Tinker tokenizers disagree")
        # Probe: one example, one step.
        probe = [train[0]]
        fwd = client.forward_backward([to_datum(probe[0])], "cross_entropy")
        opt = client.optim_step(
            types.AdamParams(learning_rate=lr_at(0, total_steps, args.lr, args.warmup))
        )
        probe_out = fwd.result()
        opt.result()
    except Exception as exc:  # noqa: BLE001 - reported and stopped cleanly
        reason = "billing/credits" if _billing_problem(exc) else "error"
        _write(
            status_path,
            {
                "status": "stopped",
                "stage": "probe",
                "reason": reason,
                "error": str(exc)[:500],
            },
        )
        print(
            f"Probe step failed ({reason}): {type(exc).__name__}: {str(exc)[:300]}",
            flush=True,
        )
        return 3
    probe_loss, _ = batch_losses(probe_out, probe)
    used_tokens = len(probe[0]["input"])
    print(f"probe ok: loss/token {probe_loss:.3f}", flush=True)

    def evaluate() -> float:
        nonlocal used_tokens
        output = client.forward([to_datum(e) for e in val], "cross_entropy").result()
        used_tokens += n_tokens(val)
        return batch_losses(output, val)[1]

    log = log_path.open("w", encoding="utf-8")
    best = (math.inf, -1, None)  # (val loss, step, sampler path)
    bad_evals, step, last_train = 0, 1, (math.nan, math.nan)
    stop = False

    def checkpoint(val_loss: float) -> None:
        nonlocal best, bad_evals
        if val_loss < best[0] - 1e-4:
            saved = client.save_weights_for_sampler(
                name=f"crux-predictor-step{step:04d}"
            ).result()
            best = (val_loss, step, saved.path)
            bad_evals = 0
        else:
            bad_evals += 1

    val_loss = evaluate()
    print(f"step 0 val_loss {val_loss:.4f}", flush=True)
    log.write(json.dumps({"step": 0, "val_loss": val_loss}) + "\n")
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch):
            batch = [train[i] for i in order[start : start + args.batch]]
            lr = lr_at(step, total_steps, args.lr, args.warmup)
            fwd = client.forward_backward([to_datum(e) for e in batch], "cross_entropy")
            opt = client.optim_step(types.AdamParams(learning_rate=lr))
            output = fwd.result()
            opt.result()
            used_tokens += n_tokens(batch)
            last_train = batch_losses(output, batch)
            entry = {
                "step": step,
                "epoch": epoch,
                "lr": lr,
                "train_loss_per_token": last_train[0],
                "train_loss": last_train[1],
                "est_usd": round(
                    used_tokens * TRAIN_PRICE.get(args.model, 1.0) / 1e6, 4
                ),
            }
            if step % args.eval_every == 0:
                val_loss = evaluate()
                checkpoint(val_loss)
                entry["val_loss"] = val_loss
                entry["best_val_loss"] = best[0]
                stop = bad_evals >= args.patience
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(
                f"step {step} ep {epoch} lr {lr:.2e} train {last_train[1]:.4f}"
                + (f" val {entry['val_loss']:.4f}" if "val_loss" in entry else ""),
                flush=True,
            )
            step += 1
            if stop:
                break
        if stop:
            print(
                f"early stop: {args.patience} evaluations without improvement",
                flush=True,
            )
            break
    step -= 1  # the last completed step
    if step % args.eval_every != 0:
        val_loss = evaluate()
        checkpoint(val_loss)
        log.write(
            json.dumps({"step": step, "val_loss": val_loss, "final": True}) + "\n"
        )
        print(f"final val_loss {val_loss:.4f}", flush=True)
    log.close()
    usd = used_tokens * TRAIN_PRICE.get(args.model, 1.0) / 1e6
    add_to_ledger(out, "train", usd, {"tokens": used_tokens, "model": args.model})
    model = {
        "model_path": best[2],
        "base_model": args.model,
        "rank": args.rank,
        "learning_rate": args.lr,
        "batch_size": args.batch,
        "epochs_planned": args.epochs,
        "steps_run": step,
        "best_step": best[1],
        "best_val_loss": best[0],
        "final_train_loss": last_train[1],
        "final_train_loss_per_token": last_train[0],
        "train_examples": len(train),
        "train_cases": len({r["case_id"] for r in train_records}),
        "val_examples": len(val),
        "tokens_processed": used_tokens,
        "est_usd": round(usd, 3),
        "seconds": round(time.monotonic() - started, 1),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write(out / "model.json", model)
    _write(status_path, {"status": "done", "model_path": best[2]})
    print(json.dumps(model, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
