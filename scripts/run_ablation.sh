#!/bin/bash
# N-step ablation study for DrQ-v2
# Tasks: cartpole_swingup, walker_walk
# N-step: 1, 3, 5, 10
# Seeds: 1, 2, 3

set -e

# ---- Configuration ----
TASKS=("cartpole_swingup" "walker_walk")
NSTEPS=(1 3 5 10)
SEEDS=(1 2 3)

declare -A TASK_FRAMES
TASK_FRAMES[cartpole_swingup]=500000
TASK_FRAMES[walker_walk]=1000000

PYTHON="${PYTHON:-d:/CSC415_Project/drqv2/venv/Scripts/python.exe}"
DRY_RUN=false

# ---- Parse arguments ----
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
    esac
done

# ---- Calculate totals ----
TOTAL_RUNS=$(( ${#TASKS[@]} * ${#NSTEPS[@]} * ${#SEEDS[@]} ))
echo "============================================"
echo "  DrQ-v2 N-step Ablation Study"
echo "============================================"
echo "Tasks:   ${TASKS[*]}"
echo "N-steps: ${NSTEPS[*]}"
echo "Seeds:   ${SEEDS[*]}"
echo "Total runs: $TOTAL_RUNS"
echo ""
for task in "${TASKS[@]}"; do
    echo "  $task: ${TASK_FRAMES[$task]} frames"
done
echo "============================================"
if $DRY_RUN; then
    echo "[DRY RUN] Only printing commands, not executing."
    echo ""
fi

# ---- Run experiments ----
RUN_IDX=0
for task in "${TASKS[@]}"; do
    frames=${TASK_FRAMES[$task]}
    for n in "${NSTEPS[@]}"; do
        for s in "${SEEDS[@]}"; do
            RUN_IDX=$((RUN_IDX + 1))
            EXP_NAME="${task}_n${n}_s${s}"

            CMD="\"$PYTHON\" train.py task@_global_=$task nstep=$n seed=$s num_train_frames=$frames experiment=$EXP_NAME save_video=false save_train_video=false replay_buffer_num_workers=1"

            echo "[$RUN_IDX/$TOTAL_RUNS] $EXP_NAME"

            if $DRY_RUN; then
                echo "  CMD: $CMD"
                echo ""
            else
                echo "  Starting..."
                eval $CMD
                echo "  Done."
                echo ""
            fi
        done
    done
done

echo "============================================"
if $DRY_RUN; then
    echo "Dry run complete. $TOTAL_RUNS commands printed."
else
    echo "All $TOTAL_RUNS runs complete."
fi
echo "============================================"
