# HMM Research Scaffold

This scaffold is offline research code only. It does not change live trading behavior and does not wire HMM regimes into live decisions.

The HMM is a market-condition estimator, not the probability engine. Its training matrix is restricted to A and D features:

- A: BTC path, volatility, and path-shape features.
- D: microstructure, spectral entropy, transition entropy, and simple magnitude-state baselines.

B, F, and H fields remain outside the HMM. Quote/executable market features, probability engine outputs, model-vs-market edge, tau/time policy, strike, outcome, and inventory fields are included only for replay analysis and visualization.

Expected growth is implemented as a hard offline replay veto. Tail/polarization and reversal vetoes remain default safety vetoes in replay. HMM-conditioned override cells are reports only; they do not permit live overrides.

Inventory management remains standalone and is not an HMM input. Position reevaluation is not used as an HMM feature.

All replay behavior is causal: training uses only walk-forward training windows, scalers are fitted only on each training fold, test posteriors are filtered forward through the test stream, and no full-sequence Viterbi labels or smoothed posteriors are used for decision fields.

Visualizations are generated from real replay/training output and are written under `artifacts/hmm_visuals/`. Research datasets and reports are written under `artifacts/hmm_research/`.

Example commands:

```bash
python scripts/build_hmm_replay_dataset.py \
  --klines data/btc_1m.csv \
  --decision-log artifacts/decision_state.jsonl \
  --output artifacts/hmm_research/replay_dataset.csv

python scripts/run_hmm_walk_forward_replay.py \
  --input artifacts/hmm_research/replay_dataset.csv \
  --output-dir artifacts/hmm_research/run_001 \
  --train-window-days 14 \
  --test-window-days 3 \
  --n-states 4

python scripts/report_hmm_policy_replay.py \
  --input artifacts/hmm_research/run_001/hmm_walk_forward_output.csv \
  --output-dir artifacts/hmm_research/run_001/policy_report

python scripts/plot_hmm_regime_overlay.py \
  --input artifacts/hmm_research/run_001/hmm_walk_forward_output.csv \
  --output artifacts/hmm_visuals/run_001_overlay.html
```

