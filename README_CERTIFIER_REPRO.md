# Certifier Reproducibility README

This README covers how to reproduce the CRC/LTT certifier experiments.

## Core Scripts

| Script | Purpose |
|--------|---------|
| `crc_core.py` | Shared certifier core: calibration, thresholding, scoring, serialization |
| `run_crc_experiment.py` | One-shot CRC fit/eval run; writes `run_args.json` |
| `run_crc_feature_alpha_sweep.py` | CRC alpha sweep on a fixed split/index |
| `run_crc_fixed_alpha_tau_sweep.py` | CRC tau sweep at fixed alpha |
| `run_ltt_feature_alpha_sweep.py` | LTT alpha sweep with feature/model options |
| `sparse_rectify.py` | Rectifier helper (MBV and sparse-L1 backends) |

## Typical Flow

1. Run `run_crc_experiment.py` to create `run_args.json` and an initial certifier.
2. Reuse that run directory for CRC and LTT alpha/tau sweeps.

## Environment

Required Python modules: `numpy`, `scipy`, `sklearn`, `hnswlib`

Install with:
```bash
python -m pip install -e .
pip install scipy scikit-learn
```

## Required Data Files

Dataset files:
- `sift_base.fvecs`, `sift_query.fvecs`, `sift_groundtruth.ivecs`
- `gist_base.fvecs`, `gist_query.fvecs`, `gist_groundtruth.ivecs`
- `deep1m_base.fbin`, `deep.query.public.10K.fbin`, `deep1M_groundtruth.ivecs`

Index files under `indexes/`:
- `sift_base_M16_efc100_hnsw.bin`
- `gist_base_M16_efc100_hnsw_changebase.bin`
- `deep1m_M16_efc100_hnsw.bin`

## Example Commands

```bash
# Base CRC run
python run_crc_experiment.py \
  --fvec-base-file sift_base.fvecs \
  --fvec-query-file sift_query.fvecs \
  --fvec-gt-file sift_groundtruth.ivecs \
  --M 16 --efc 100 --efs 100 --k 100 \
  --tau 0.90 --alpha 0.10 \
  --cal-size 5000 --qcnt 10000 --seed 0 \
  --out-dir crc_runs/sift_crc_base

# CRC alpha sweep
python run_crc_feature_alpha_sweep.py \
  --run-dir crc_runs/sift_crc_base \
  --out-dir crc_runs/sift_crc_alpha_sweep \
  --tau 0.90 --alpha-mode adaptive

# CRC tau sweep at fixed alpha
python run_crc_fixed_alpha_tau_sweep.py \
  --run-dir crc_runs/sift_crc_base \
  --out-dir crc_runs/sift_crc_tau_sweep \
  --fixed-alpha 0.10

# LTT alpha sweep
python run_ltt_feature_alpha_sweep.py \
  --run-dir crc_runs/sift_crc_base \
  --out-dir crc_runs/sift_ltt_alpha_sweep \
  --tau 0.90 --epsilon 0.05 \
  --alphas 0.1,0.05,0.01,0.005
```

## Output Artifacts

Each sweep writes:
- `run_args.json` — experiment configuration for reproducibility
- `certifier.json` — fitted certifier (CRC runs)
- `metrics.json` — summary evaluation metrics
- `*_summary.csv` — per-alpha/tau aggregate results
- `*_per_query.csv` — per-query detail rows
