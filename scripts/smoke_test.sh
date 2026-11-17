#!/usr/bin/env bash
# Quick end-to-end check of the install (a few minutes on CPU).
# Runs a tiny SAC job on cheetah-run that also trains the conditional diffusion model once
# and samples synthetic transitions from it, so every moving part gets exercised.
#   bash scripts/smoke_test.sh                 # auto device (cuda if available, else cpu)
#   bash scripts/smoke_test.sh --device mps    # try Apple's GPU backend
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${TMPDIR:-/tmp}/pgr_smoke_results"
mkdir -p "$OUT"

python synther/online/online_cond.py \
  --env cheetah-run-v0 \
  --gin_config_files config/online/sac_cond_synther_dmc.gin \
  --gin_params \
    'redq_sac.epochs = 3' \
    'redq_sac.steps_per_epoch = 100' \
    'redq_sac.max_ep_len = 100' \
    'redq_sac.start_steps = 100' \
    'redq_sac.utd_ratio = 2' \
    'redq_sac.n_evals_per_epoch = 1' \
    'redq_sac.replay_size = 20000' \
    'redq_sac.diffusion_buffer_size = 5000' \
    'redq_sac.retrain_diffusion_every = 200' \
    'redq_sac.num_samples = 1000' \
    'REDQCondTrainer.train_num_steps = 20' \
    'CondDiffusionGenerator.sample_batch_size = 1000' \
    'CondDiffusionGenerator.num_sample_steps = 8' \
  --log_dir smoke_test \
  --results_folder "$OUT" \
  "$@"

echo "SMOKE TEST PASSED"
