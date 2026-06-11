# Certifier Quickstart (CRC/LTT)

Minimal guide for running certifier sweeps.

## Core Files
- `crc_core.py`: shared certifier core (calibration, thresholding, scoring, serialization).
- `run_crc_experiment.py`: one-shot CRC fit/eval run that writes `run_args.json`.
- `run_crc_feature_alpha_sweep.py`: CRC alpha sweep on a fixed split/index.
- `run_crc_fixed_alpha_tau_sweep.py`: CRC tau sweep at fixed alpha.
- `run_ltt_feature_alpha_sweep.py`: LTT alpha sweep with feature/model options.

## Typical Flow
1. Run a base CRC experiment to create `run_args.json`.
2. Reuse that run directory for CRC and LTT alpha/tau sweeps.

## Example Commands
```bash
# 1) Base CRC run
python run_crc_experiment.py \
  --fvec-base-file sift_base.fvecs \
  --fvec-query-file sift_query.fvecs \
  --fvec-gt-file sift_groundtruth.ivecs \
  --M 16 --efc 100 --efs 100 --k 100 \
  --tau 0.90 --alpha 0.10 \
  --cal-size 5000 --qcnt 10000 --seed 0 \
  --out-dir crc_runs/sift_crc_base

# 2) CRC alpha sweep on that split/index
python run_crc_feature_alpha_sweep.py \
  --run-dir crc_runs/sift_crc_base \
  --out-dir crc_runs/sift_crc_alpha_sweep \
  --tau 0.90 --alpha-mode adaptive

# 3) CRC fixed-alpha tau sweep
python run_crc_fixed_alpha_tau_sweep.py \
  --run-dir crc_runs/sift_crc_base \
  --out-dir crc_runs/sift_crc_tau_sweep \
  --fixed-alpha 0.10

# 4) LTT alpha sweep on the same split/index
python run_ltt_feature_alpha_sweep.py \
  --run-dir crc_runs/sift_crc_base \
  --out-dir crc_runs/sift_ltt_alpha_sweep \
  --tau 0.90 --epsilon 0.05 \
  --alphas 0.1,0.05,0.01,0.005
```

## Output Artifacts
- `run_args.json`
- `certifier.json` (CRC)
- `metrics.json`
- sweep summary/per-query CSV files
