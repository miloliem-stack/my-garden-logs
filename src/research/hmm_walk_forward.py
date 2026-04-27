from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .hmm_dataset import read_table, write_manifest, write_table
from .hmm_features import HMM_FEATURE_COLUMNS, add_training_entropy_percentiles


@dataclass
class DiagGaussianHMM:
    n_states: int
    random_seed: int = 42
    covariance_floor: float = 1e-4
    startprob_: np.ndarray | None = None
    transmat_: np.ndarray | None = None
    means_: np.ndarray | None = None
    covars_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "DiagGaussianHMM":
        if len(x) < self.n_states:
            raise ValueError("not enough training rows for requested HMM states")
        labels = KMeans(n_clusters=self.n_states, random_state=self.random_seed, n_init=10).fit_predict(x)
        self.startprob_ = np.bincount(labels[:1], minlength=self.n_states).astype(float) + 1.0
        self.startprob_ /= self.startprob_.sum()
        trans = np.ones((self.n_states, self.n_states), dtype=float)
        for prev, cur in zip(labels[:-1], labels[1:]):
            trans[prev, cur] += 1.0
        self.transmat_ = trans / trans.sum(axis=1, keepdims=True)
        global_var = np.var(x, axis=0) + self.covariance_floor
        means = []
        covars = []
        for state in range(self.n_states):
            sample = x[labels == state]
            if len(sample) == 0:
                means.append(np.mean(x, axis=0))
                covars.append(global_var)
            else:
                means.append(np.mean(sample, axis=0))
                covars.append(np.maximum(np.var(sample, axis=0), self.covariance_floor))
        self.means_ = np.asarray(means, dtype=float)
        self.covars_ = np.asarray(covars, dtype=float)
        return self

    def emission_prob(self, obs: np.ndarray) -> np.ndarray:
        diff = obs[None, :] - self.means_
        logp = -0.5 * (np.sum(np.log(2.0 * np.pi * self.covars_), axis=1) + np.sum((diff * diff) / self.covars_, axis=1))
        logp -= np.max(logp)
        prob = np.exp(logp)
        return prob / max(prob.sum(), 1e-300)

    def filter_stream(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        posteriors = []
        next_probs = []
        alpha = np.asarray(self.startprob_, dtype=float)
        for obs in x:
            pred = alpha @ self.transmat_
            emission = self.emission_prob(obs)
            alpha = pred * emission
            alpha = alpha / max(alpha.sum(), 1e-300)
            posteriors.append(alpha.copy())
            next_probs.append(alpha @ self.transmat_)
        return np.asarray(posteriors), np.asarray(next_probs)


def _clean_feature_frame(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    x = df.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan)
    return x.ffill().fillna(0.0)


def _folds_by_dates(df: pd.DataFrame, args: dict) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if args.get("train_start") and args.get("train_end") and args.get("test_start") and args.get("test_end"):
        return [tuple(pd.Timestamp(args[k], tz="UTC") for k in ["train_start", "train_end", "test_start", "test_end"])]
    train_days = int(args.get("train_window_days") or 14)
    test_days = int(args.get("test_window_days") or 3)
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    folds = []
    train_start = ts_min
    while True:
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_start > ts_max:
            break
        folds.append((train_start, train_end, test_start, min(test_end, ts_max)))
        train_start = test_end
    return folds


def _regime_states(post: np.ndarray, next_prob: np.ndarray, *, confidence_threshold: float, next_same_threshold: float, persistence_threshold: int) -> tuple[list[str], list[int]]:
    labels = []
    counts = []
    last_state = None
    persistence = 0
    for p, n in zip(post, next_prob):
        state = int(np.argmax(p))
        if state == last_state:
            persistence += 1
        else:
            persistence = 1
            last_state = state
        confidence = float(p[state])
        next_same = float(n[state])
        if confidence < confidence_threshold or next_same < next_same_threshold or persistence < persistence_threshold:
            labels.append("transition_uncertain")
        else:
            labels.append(f"state_{state}_confident")
        counts.append(persistence)
    return labels, counts


def run_walk_forward(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    n_states: int = 4,
    covariance_type: str = "diag",
    feature_columns: Sequence[str] | None = None,
    random_seed: int = 42,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    train_window_days: int | None = None,
    test_window_days: int | None = None,
    confidence_threshold: float = 0.70,
    next_same_threshold: float = 0.65,
    persistence_threshold: int = 2,
) -> pd.DataFrame:
    if covariance_type != "diag":
        raise ValueError("only diagonal covariance is supported by the local offline HMM")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_table(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    feature_columns = list(feature_columns or HMM_FEATURE_COLUMNS)
    folds = _folds_by_dates(
        df,
        {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_window_days": train_window_days,
            "test_window_days": test_window_days,
        },
    )
    all_rows = []
    metadata = []
    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        train_mask = (df["timestamp"] >= tr_start) & (df["timestamp"] <= tr_end)
        test_mask = (df["timestamp"] >= te_start) & (df["timestamp"] <= te_end)
        fold_df = add_training_entropy_percentiles(df, train_mask=train_mask)
        train = fold_df.loc[train_mask].copy()
        test = fold_df.loc[test_mask].copy()
        if train.empty or test.empty:
            continue
        train_x = _clean_feature_frame(train, feature_columns)
        test_x = _clean_feature_frame(test, feature_columns)
        scaler = StandardScaler().fit(train_x)
        model = DiagGaussianHMM(n_states=n_states, random_seed=random_seed).fit(scaler.transform(train_x))
        post, next_prob = model.filter_stream(scaler.transform(test_x))
        labels, persistence = _regime_states(
            post,
            next_prob,
            confidence_threshold=confidence_threshold,
            next_same_threshold=next_same_threshold,
            persistence_threshold=persistence_threshold,
        )
        result = test.copy()
        result["fold_id"] = fold_idx
        for state in range(n_states):
            result[f"hmm_state_prob_{state}"] = post[:, state]
            result[f"hmm_next_state_prob_{state}"] = next_prob[:, state]
        result["hmm_map_state"] = np.argmax(post, axis=1)
        result["hmm_map_confidence"] = np.max(post, axis=1)
        result["hmm_entropy"] = -np.sum(post * np.log(post + 1e-12), axis=1) / np.log(n_states)
        result["hmm_next_same_state_confidence"] = [next_prob[i, int(result["hmm_map_state"].iloc[i])] for i in range(len(result))]
        result["hmm_map_state_persistence_count"] = persistence
        result["regime_policy_state"] = labels
        all_rows.append(result)
        state_counts = result["hmm_map_state"].value_counts().sort_index().to_dict()
        transitions = pd.crosstab(result["hmm_map_state"].shift(1), result["hmm_map_state"]).to_dict()
        metadata.append(
            {
                "fold_id": fold_idx,
                "train_period": [tr_start.isoformat(), tr_end.isoformat()],
                "test_period": [te_start.isoformat(), te_end.isoformat()],
                "feature_columns": feature_columns,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "hmm_transition_matrix": model.transmat_.tolist(),
                "hmm_means": model.means_.tolist(),
                "hmm_covariances": model.covars_.tolist(),
                "state_duration_summary": {str(k): int(v) for k, v in state_counts.items()},
                "transition_count_summary": transitions,
            }
        )
    if not all_rows:
        raise ValueError("no walk-forward folds produced output")
    result_df = pd.concat(all_rows, ignore_index=True)
    write_table(result_df, out_dir / "hmm_walk_forward_output.csv", fmt="csv")
    write_manifest(out_dir / "fold_metadata.json", {"folds": metadata})
    return result_df

