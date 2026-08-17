"""Phase 1 empirical comparison of mutation GA and CMA-ES-with-margin."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import optuna
import pandas as pd
import yfinance as yf

TICKERS = ("SPY", "TLT", "TIP", "SGOV", "GLD", "DBC", "DBMF", "VNQ")
FACTORS = {
    "SPY": "equity",
    "TLT": "nominal_duration",
    "TIP": "real_duration",
    "SGOV": "cash",
    "GLD": "gold",
    "DBC": "commodities",
    "DBMF": "managed_futures",
    "VNQ": "real_estate",
}
BLOCK_SIZE = 21
BOOTSTRAPS = 24
INITIAL_VALUE = 100_000.0
MONTHLY_CONTRIBUTION = 1_000.0


@dataclass(frozen=True)
class Candidate:
    tickers: tuple[str, ...]
    weights: tuple[int, ...]

    def key(self) -> str:
        return ",".join(f"{ticker}:{weight}" for ticker, weight in zip(self.tickers, self.weights))


def load_returns(start: str, end: str) -> pd.DataFrame:
    """Download one aligned daily-return matrix for the bake-off slice."""
    prices = yf.download(
        list(TICKERS),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices["Close"]
    else:
        prices = prices[["Close"]].rename(columns={"Close": TICKERS[0]})
    prices = prices.reindex(columns=TICKERS).dropna(how="any")
    returns = prices.pct_change().dropna()
    if len(returns) < 252:
        raise RuntimeError(f"Only {len(returns)} aligned daily returns were downloaded")
    return returns


def _monthly_cashflows(index: pd.DatetimeIndex) -> np.ndarray:
    month = index.to_period("M")
    return np.r_[INITIAL_VALUE, np.where(month[1:] != month[:-1], MONTHLY_CONTRIBUTION, 0.0)]


def mwrr(returns: np.ndarray, index: pd.DatetimeIndex) -> float:
    """Annualized money-weighted return for daily returns and monthly deposits."""
    values = np.empty(len(returns) + 1, dtype=float)
    values[0] = INITIAL_VALUE
    deposits = _monthly_cashflows(index)
    for position, daily_return in enumerate(returns, start=1):
        values[position] = (values[position - 1] + deposits[position - 1]) * (1.0 + daily_return)
    dates = np.r_[0.0, (index - index[0]).days / 365.25]
    cashflows = np.r_[-deposits, 0.0]
    cashflows[-1] += values[-1]
    if np.all(cashflows[:-1] <= 0) and cashflows[-1] > 0:
        low, high = -0.999, 10.0
        for _ in range(80):
            rate = (low + high) / 2.0
            npv = np.sum(cashflows / np.power(1.0 + rate, dates))
            if npv > 0:
                low = rate
            else:
                high = rate
        return (low + high) / 2.0
    return -1.0


def block_bootstrap(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Circular block-bootstrap one synthetic history of equal length."""
    starts = rng.integers(0, len(returns), size=(len(returns) + BLOCK_SIZE - 1) // BLOCK_SIZE)
    blocks = [returns[(start + np.arange(BLOCK_SIZE)) % len(returns)] for start in starts]
    return np.concatenate(blocks)[: len(returns)]


def portfolio_returns(returns: pd.DataFrame, candidate: Candidate) -> np.ndarray:
    weights = np.asarray(candidate.weights, dtype=float) / 100.0
    return returns.loc[:, list(candidate.tickers)].to_numpy() @ weights


def fitness(
    returns: pd.DataFrame,
    candidate: Candidate,
    seed: int,
    bootstraps: int = BOOTSTRAPS,
) -> float:
    """Score mean bootstrap MWRR, penalizing dispersion and isolated peaks."""
    history = portfolio_returns(returns, candidate)
    rng = np.random.default_rng(seed)
    scores = np.array([mwrr(block_bootstrap(history, rng), returns.index) for _ in range(bootstraps)])
    return float(scores.mean() - 0.25 * scores.std())


def weight_vectors() -> Iterable[tuple[int, int, int]]:
    for first in range(10, 81, 10):
        for second in range(10, 91 - first, 10):
            third = 100 - first - second
            if third >= 10:
                yield first, second, third


def enumerate_candidates() -> list[Candidate]:
    return [
        Candidate(tickers, weights)
        for tickers in itertools.combinations(TICKERS, 3)
        for weights in weight_vectors()
    ]


def ground_truth(returns: pd.DataFrame, seed: int) -> tuple[Candidate, dict[str, float]]:
    scores = {candidate.key(): fitness(returns, candidate, seed) for candidate in enumerate_candidates()}
    best_key = max(scores, key=scores.get)
    best = next(candidate for candidate in enumerate_candidates() if candidate.key() == best_key)
    return best, scores


def mutate(candidate: Candidate, rng: np.random.Generator) -> Candidate:
    weights = list(candidate.weights)
    if rng.random() < 0.7:
        left, right = sorted(rng.choice(3, size=2, replace=False))
        delta = int(rng.choice([-10, 10]))
        if weights[left] + delta >= 10 and weights[right] - delta >= 10:
            weights[left] += delta
            weights[right] -= delta
    else:
        ticker_pos = int(rng.integers(0, 3))
        replacement = rng.choice([ticker for ticker in TICKERS if ticker not in candidate.tickers])
        tickers = list(candidate.tickers)
        tickers[ticker_pos] = str(replacement)
        return Candidate(tuple(sorted(tickers)), tuple(weights))
    return Candidate(candidate.tickers, tuple(weights))


def run_ga(returns: pd.DataFrame, budget: int, seed: int) -> tuple[Candidate, int]:
    rng = np.random.default_rng(seed)
    candidates = enumerate_candidates()
    current = candidates[int(rng.integers(len(candidates)))]
    best = current
    best_score = float("-inf")
    seen = 0
    while seen < budget:
        score = fitness(returns, current, seed + seen)
        seen += 1
        if score > best_score:
            best, best_score = current, score
        current = mutate(best, rng)
    return best, seen


def candidate_from_vector(values: list[float]) -> Candidate:
    positions = np.argsort(values)[-3:]
    raw = np.asarray([values[position] for position in positions])
    scaled = np.maximum(1, np.rint(raw / raw.sum() * 10).astype(int))
    while scaled.sum() != 10:
        position = int(np.argmax(scaled)) if scaled.sum() > 10 else int(np.argmin(scaled))
        if scaled.sum() > 10 and scaled[position] > 1:
            scaled[position] -= 1
        elif scaled.sum() < 10:
            scaled[position] += 1
    tickers = tuple(sorted(TICKERS[position] for position in positions))
    ordered = tuple(int(weight * 10) for _, weight in sorted(zip((TICKERS[position] for position in positions), scaled)))
    return Candidate(tickers, ordered)


def run_cmaes(returns: pd.DataFrame, budget: int, seed: int) -> tuple[Candidate, int]:
    sampler = optuna.samplers.CmaEsSampler(seed=seed, with_margin=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        values = [trial.suggest_float(f"weight_{position}", 0.001, 1.0) for position in range(len(TICKERS))]
        candidate = candidate_from_vector(values)
        return fitness(returns, candidate, seed + trial.number)

    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return candidate_from_vector([study.best_trial.params[f"weight_{position}"] for position in range(len(TICKERS))]), budget


@dataclass(frozen=True)
class Result:
    algorithm: str
    candidate: str
    evaluations: int
    ground_truth_score: float
    candidate_score: float
    ticker_overlap: int


def run_bakeoff(start: str, end: str, budget: int, seed: int) -> dict[str, object]:
    returns = load_returns(start, end)
    truth, truth_scores = ground_truth(returns, seed)
    ga, ga_evaluations = run_ga(returns, budget, seed + 10_000)
    cma, cma_evaluations = run_cmaes(returns, budget, seed + 20_000)
    results = []
    for algorithm, candidate, evaluations in (("mutation_ga", ga, ga_evaluations), ("cmaes_margin", cma, cma_evaluations)):
        results.append(
            asdict(
                Result(
                    algorithm,
                    candidate.key(),
                    evaluations,
                    truth_scores[truth.key()],
                    truth_scores.get(candidate.key(), fitness(returns, candidate, seed)),
                    len(set(candidate.tickers) & set(truth.tickers)),
                )
            )
        )
    return {
        "tickers": list(TICKERS),
        "factors": FACTORS,
        "date_range": {"start": str(returns.index[0].date()), "end": str(returns.index[-1].date())},
        "aligned_observations": len(returns),
        "subset_size": 3,
        "weight_grid_percent": 10,
        "ground_truth_candidates": len(truth_scores),
        "ground_truth": {"candidate": truth.key(), "score": truth_scores[truth.key()]},
        "evaluation_budget": budget,
        "bootstrap_histories_per_fitness": BOOTSTRAPS,
        "block_size": BLOCK_SIZE,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", type=Path, default=Path("artifacts/phase1_bakeoff.json"))
    args = parser.parse_args()
    result = run_bakeoff(args.start, args.end, args.budget, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
