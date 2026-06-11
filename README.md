# HNSW with Accuracy Guarantees

This repository contains the code for the paper experiments on CRC/LTT-based recall certification for HNSW.

It keeps the paper-relevant code paths:

- `hnswlib/`, `python_bindings/`: augmented C++ core and Python bindings
- `crc_core.py`: shared certifier core (calibration, thresholding, scoring, serialization)
- `run_crc_experiment.py`: one-shot CRC certifier runner
- `run_crc_feature_alpha_sweep.py`: CRC alpha sweep on a fixed split/index
- `run_crc_fixed_alpha_tau_sweep.py`: CRC tau sweep at fixed alpha
- `run_ltt_feature_alpha_sweep.py`: LTT alpha sweep with feature/model options
- `sparse_rectify.py`: rectifier helper (MBV / sparse-L1 backends)
- `scripts/reproduce_certifier_suite.sh`: environment check helper
- `paper/recall_certification.pdf`: companion paper

## Quick Start

Build the Python extension:

```bash
python -m pip install -e .
```

The experiments expect external dataset and index files:

- `sift_base.fvecs`, `sift_query.fvecs`, `sift_groundtruth.ivecs`
- `gist_base.fvecs`, `gist_query.fvecs`, `gist_groundtruth.ivecs`
- `deep1m_base.fbin`, `deep.query.public.10K.fbin`, `deep1M_groundtruth.ivecs`
- HNSW index files under `indexes/`

## Running the Certifier Experiments

```bash
# 1) Validate environment and inputs
./scripts/reproduce_certifier_suite.sh check

# 2) Base CRC run
python run_crc_experiment.py \
  --fvec-base-file sift_base.fvecs \
  --fvec-query-file sift_query.fvecs \
  --fvec-gt-file sift_groundtruth.ivecs \
  --M 16 --efc 100 --efs 100 --k 100 \
  --tau 0.90 --alpha 0.10 \
  --cal-size 5000 --qcnt 10000 --seed 0 \
  --out-dir crc_runs/sift_crc_base

# 3) CRC alpha sweep on that split/index
python run_crc_feature_alpha_sweep.py \
  --run-dir crc_runs/sift_crc_base \
  --out-dir crc_runs/sift_crc_alpha_sweep \
  --tau 0.90 --alpha-mode adaptive

# 4) LTT alpha sweep on the same split/index
python run_ltt_feature_alpha_sweep.py \
  --run-dir crc_runs/sift_crc_base \
  --out-dir crc_runs/sift_ltt_alpha_sweep \
  --tau 0.90 --epsilon 0.05 \
  --alphas 0.1,0.05,0.01,0.005
```

## Notes

- Generated CSV outputs are not committed; rerun scripts will recreate them into local output directories.
