#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
MODE="${1:-check}"

log() {
  printf '[repro] %s\n' "$*"
}

die() {
  printf '[repro][error] %s\n' "$*" >&2
  exit 1
}

run() {
  log "run: $*"
  "$@"
}

require_file() {
  local p="$1"
  [[ -f "$p" ]] || die "missing file: $p"
}

require_dir() {
  local p="$1"
  [[ -d "$p" ]] || die "missing dir: $p"
}

check_env() {
  [[ -x "$PYTHON_BIN" ]] || die "python not executable: $PYTHON_BIN"

  run "$PYTHON_BIN" - <<'PY'
import importlib
mods = ["numpy", "scipy", "sklearn", "hnswlib"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing python modules in environment: {missing}")
print("python env ok")
PY
}

check_inputs() {
  require_dir "$ROOT/indexes"

  # Dataset vectors / GT.
  require_file "$ROOT/sift_base.fvecs"
  require_file "$ROOT/sift_query.fvecs"
  require_file "$ROOT/sift_groundtruth.ivecs"
  require_file "$ROOT/gist_base.fvecs"
  require_file "$ROOT/gist_query.fvecs"
  require_file "$ROOT/gist_groundtruth.ivecs"
  require_file "$ROOT/deep1m_base.fbin"
  require_file "$ROOT/deep.query.public.10K.fbin"
  require_file "$ROOT/deep1M_groundtruth.ivecs"

  # M16 indexes used by the alpha sweep scripts.
  require_file "$ROOT/indexes/sift_base_M16_efc100_hnsw.bin"
  require_file "$ROOT/indexes/gist_base_M16_efc100_hnsw_changebase.bin"
  require_file "$ROOT/indexes/deep1m_M16_efc100_hnsw.bin"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/reproduce_certifier_suite.sh [mode]

Modes:
  check  Validate environment and required inputs (default).

Example alpha sweep runs:
  # CRC base run (creates run_args.json)
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
    --tau 0.90

  # LTT alpha sweep
  python run_ltt_feature_alpha_sweep.py \
    --run-dir crc_runs/sift_crc_base \
    --out-dir crc_runs/sift_ltt_alpha_sweep \
    --tau 0.90 --alphas 0.1,0.05,0.01
EOF
}

case "$MODE" in
  check)
    check_env
    check_inputs
    log "check completed"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    die "unknown mode: $MODE"
    ;;
esac
