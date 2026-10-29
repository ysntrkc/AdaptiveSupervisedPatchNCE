#!/usr/bin/env bash
# One round of ASP (paper settings from experiments/mist_launcher.py) on BCI, MIST HER2 and HER2match:
# train -> test on the full 1024 image -> evaluate with BCI's evaluate.py (same metric code for every method).
#
# Usage: scripts/baseline_round.sh [variant] [datasets...]
#   defaults: baseline "BCI MIST/HER2 HER2match"   (reads <dataset>/asp/{trainA,trainB,testA,testB})
# Run detached:  setsid nohup scripts/baseline_round.sh > /mnt/d/thesis/runs/_logs/asp_round.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

VARIANT=${1:-baseline}
shift $(( $# < 1 ? $# : 1 ))
DATASETS=${*:-BCI MIST/HER2 HER2match}
RUNS=${RUNS:-/mnt/d/thesis/runs}          # override for smoke tests
EXTRA_TRAIN=${EXTRA_TRAIN:-}               # e.g. "--max_dataset_size 4 --n_epochs 1 --n_epochs_decay 1"
EXTRA_TEST=${EXTRA_TEST:-}                 # e.g. "--num_test 4"
EVAL_DEVICE=${EVAL_DEVICE:-cuda}
DATA=/mnt/d/thesis/datasets
BCI=$HOME/workspace/tez/BCI
LOGS=$RUNS/_logs
mkdir -p "$LOGS"

# model options shared by train and test (experiments/mist_launcher.py)
MODEL_OPTS=(--model cpt --CUT_mode FastCUT --netG resnet_6blocks --netD n_layers --n_layers_D 5 --ndf 32
    --normG instance --normD instance --weight_norm spectral --lambda_GAN 1.0 --lambda_NCE 10.0
    --nce_layers 0,4,8,12,16 --nce_T 0.07 --num_patches 256 --lambda_gp 10.0
    --gp_weights '[0.015625,0.03125,0.0625,0.125,0.25,1.0]' --lambda_asp 10.0 --asp_loss_mode lambda_linear
    --dataset_mode aligned --direction AtoB --load_size 1024)

log() { echo "[$(date '+%F %T')] $*"; }

for DS in $DATASETS; do
    RUN=asp/$VARIANT/run_$(date +%Y%m%d_%H%M%S)_${DS//\//_}
    mkdir -p "$RUNS/$RUN"
    git log -1 --format='%H %s' > "$RUNS/$RUN/git_commit.txt"
    log "train $RUN"
    # shellcheck disable=SC2086
    if ! uv run python train.py --dataroot "$DATA/$DS/asp" --checkpoints_dir "$RUNS" --name "$RUN" "${MODEL_OPTS[@]}" \
            --batch_size 1 --num_threads 8 --preprocess crop --crop_size 512 --n_epochs 30 --n_epochs_decay 10 \
            --save_epoch_freq 5 --update_html_freq 100 --display_id -1 $EXTRA_TRAIN \
            > "$RUNS/$RUN/training_output.txt" 2>&1; then
        log "FAILED training $RUN, see $RUNS/$RUN/training_output.txt"; continue
    fi
    # shellcheck disable=SC2086
    if uv run python test.py --dataroot "$DATA/$DS/asp" --checkpoints_dir "$RUNS" --results_dir "$RUNS" --name "$RUN" \
            "${MODEL_OPTS[@]}" --phase test --preprocess crop --crop_size 1024 --num_test 100000 $EXTRA_TEST \
            > "$RUNS/$RUN/test_full1024.log" 2>&1 \
       && uv run --project "$BCI" python "$BCI/PyramidPix2pix/evaluate.py" --result_path "$RUNS/$RUN" \
            --images_subdir test_latest --device "$EVAL_DEVICE" > "$RUNS/$RUN/eval_full1024.log" 2>&1; then
        log "tested+evaluated $RUN [full1024]"
    else
        log "FAILED test/eval $RUN, see $RUNS/$RUN/*_full1024.log"
    fi
done

python3 ~/workspace/tez/tools/build_runs_index.py "$RUNS"
log "round done"
