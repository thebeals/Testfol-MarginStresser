"""Generation orchestration and process-safe candidate evaluation."""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .fitness import score_candidate
from .db import CandidateStore
from .diversify import check_diversification
from .rebalance import SCREENER_REBALANCE_FREQUENCIES, rebalance_returns
from .validation import run_validation_suite


@dataclass(frozen=True)
class CandidateTask:
    allocation: dict[str, float]
    rebalance_freq: str = "None"
    generation: int = 0


@dataclass(frozen=True)
class CandidateResult:
    allocation: dict[str, float]
    rebalance_freq: str
    generation: int
    fitness: float
    mwrr: float
    plateau_stability: float
    dsr: float
    confidence_badge: str
    diversification_passed: bool
    diversification: dict[str, object]


def evaluate_candidate(returns: pd.DataFrame, task: CandidateTask) -> CandidateResult:
    """Evaluate one candidate without any database or external-service writes."""
    if task.rebalance_freq not in SCREENER_REBALANCE_FREQUENCIES:
        raise ValueError(f"unsupported rebalance frequency: {task.rebalance_freq}")
    fitness = score_candidate(returns, task.allocation, seed=task.generation, rebalance_freq=task.rebalance_freq)
    portfolio = rebalance_returns(returns, task.allocation, task.rebalance_freq)
    validation = run_validation_suite(portfolio, n_trials=1, n_bootstrap=200, n_perms=200)
    diversification = check_diversification(returns, task.allocation)
    return CandidateResult(
        allocation=task.allocation,
        rebalance_freq=task.rebalance_freq,
        generation=task.generation,
        fitness=fitness.score,
        mwrr=fitness.mean_mwrr,
        plateau_stability=fitness.plateau_stability,
        dsr=validation.dsr,
        confidence_badge=validation.badge.value,
        diversification_passed=diversification.passed,
        diversification={
            "factor_variance_share": diversification.factor_variance_share,
            "stress_correlations": diversification.stress_correlations,
            "violations": list(diversification.violations),
        },
    )


def _writer_loop(queue: mp.Queue, output_path: str, db_path: str | None) -> None:
    """Single writer process; workers never open the results file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = CandidateStore(db_path) if db_path else None
    with path.open("a", encoding="utf-8") as handle:
        while True:
            item = queue.get()
            if item is None:
                return
            handle.write(json.dumps(item, sort_keys=True) + "\n")
            handle.flush()
            if store:
                store.save(item)
    if store:
        store.close()


def evaluate_candidates_parallel(
    returns: pd.DataFrame,
    tasks: Iterable[CandidateTask],
    *,
    output_path: str | Path,
    db_path: str | Path | None = None,
    workers: int | None = None,
) -> list[CandidateResult]:
    """Evaluate candidates in workers and serialize through one writer process."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    writer = context.Process(target=_writer_loop, args=(queue, str(output_path), str(db_path) if db_path else None))
    writer.start()
    results: list[CandidateResult] = []
    try:
        with context.Pool(processes=workers) as pool:
            arguments = [(returns, task) for task in tasks]
            for result in pool.starmap(evaluate_candidate, arguments):
                results.append(result)
                queue.put(asdict(result))
    finally:
        queue.put(None)
        writer.join()
    return results


def generation_stale(previous_top: tuple[str, ...] | None, current_top: tuple[str, ...], stale_generations: int) -> tuple[bool, int]:
    """Track the spec's 25-generation no-change starting heuristic."""
    changed = previous_top != current_top
    next_stale = 0 if changed else stale_generations + 1
    return next_stale >= 25, next_stale
