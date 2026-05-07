#!/bin/bash
# MT-SeqKD: off-policy SFT on a pre-merged 2-teacher dataset (no online teachers).
# Generate the merged JSONL first via ``python data/prepare_mt_seqkd.py``.
#
# Sequence-length budget (paper §5):
#   agent (ToolACE, default) : --max_length 16384 --max_completion_length 4096
#   code  (OpenThoughts)     : --max_length 4096  --max_completion_length 16384
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" MAD_OPD_AUTO_PATCH=1

STUDENT=${STUDENT:-Qwen/Qwen3-4B}
TEACHER1=${TEACHER1:-Qwen/Qwen3-14B}
DATASET=${DATASET:-./data_cache/seqkd/merged.jsonl}
OUTPUT=${OUTPUT:-./outputs/mt_seqkd}
NPROC=${NPROC:-8}
mkdir -p "$OUTPUT"

torchrun --nproc_per_node=$NPROC --master_port=29500 \
    -m mad_opd.cli.train \
    --rlhf_type gkd --gkd_algorithm mt_seqkd \
    --model "$STUDENT" --teacher_model "$TEACHER1" \
    --template qwen3 --model_type qwen3 --tuner_type full \
    --dataset "$DATASET" \
    --seq_kd false --lmbda 0 --beta 0 --enable_thinking false \
    --torch_dtype bfloat16 --num_train_epochs 1 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 --weight_decay 0.01 --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 16384 --max_completion_length 4096 \
    --attn_impl flash_attn --gradient_checkpointing true --deepspeed zero3 \
    --save_strategy steps --save_steps 5 --save_only_model true --logging_steps 1 \
    --output_dir "$OUTPUT" --report_to tensorboard "$@"
