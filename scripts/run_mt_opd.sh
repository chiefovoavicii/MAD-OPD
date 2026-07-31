#!/bin/bash
# MT-OPD: 2 teachers, no debate, fixed equal-weight JSD.
#
# Sequence-length budget (paper §5):
#   agent (ToolACE, default) : --max_length 16384 --max_completion_length 4096
#   code  (OpenThoughts)     : --max_length 4096  --max_completion_length 16384
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" MAD_OPD_AUTO_PATCH=1

STUDENT=${STUDENT:-Qwen/Qwen3-4B}
TEACHER1=${TEACHER1:-Qwen/Qwen3-14B}
TEACHER2=${TEACHER2:-Qwen/Qwen3-8B}
OUTPUT=${OUTPUT:-./outputs/mt_opd}

if [[ -z ${DATASET+x} ]]; then
    export TOOLACE_DATASET_PATH=${TOOLACE_DATASET_PATH:-./data_cache/toolace/data.jsonl}
    DATASET=toolace
    DATASET_ARGS=(--toolace_mode true --custom_register_path data/toolace_dataset.py)
else
    DATASET_ARGS=()
fi
mkdir -p "$OUTPUT"

LAUNCH=$(python scripts/launch_teachers.py --external-teachers false \
    --teacher-model-1 "$TEACHER1" --teacher-model-2 "$TEACHER2" \
    --vllm-teacher-1-host 127.0.0.1 --vllm-teacher-1-port 8001 --vllm-teacher-1-gpu 0 \
    --vllm-teacher-2-host 127.0.0.1 --vllm-teacher-2-port 8002 --vllm-teacher-2-gpu 1 \
    --vllm-teacher-1-max-model-len 32768 --vllm-teacher-2-max-model-len 32768 \
    --vllm-gpu-memory-utilization 0.45 --vllm-sidecar-coexist-gpu-util 0.45 \
    --use-sidecar true --sidecar-1-port 8081 --sidecar-2-port 8082 \
    --sidecar-1-gpu 0 --sidecar-2-gpu 1 --sidecar-max-length 40960 \
    --sidecar-coexist-gpu-memory-fraction 0.45 --sidecar-enable-thinking false \
    --log-dir "$OUTPUT/teacher_logs")
get() { echo "$LAUNCH" | python -c "import sys,json; print(json.load(sys.stdin).get('$1',''))"; }
T1_URL=$(get teacher_1_url) T2_URL=$(get teacher_2_url)
S1_URL=$(get sidecar_1_url) S2_URL=$(get sidecar_2_url)
PIDS=$(echo "$LAUNCH" | python -c "import sys,json;print(' '.join(map(str,json.load(sys.stdin).get('pids',[]))))")
trap "kill -9 $PIDS 2>/dev/null || true" EXIT
export CUDA_VISIBLE_DEVICES=$(get cuda_visible_devices)

torchrun --nproc_per_node=$(get train_nproc_per_node) --master_port=29500 \
    -m mad_opd.cli.train \
    --rlhf_type gkd --gkd_algorithm mt_opd \
    --model "$STUDENT" --teacher_model "$TEACHER1" --teacher_model_2 "$TEACHER2" \
    --template qwen3 --model_type qwen3 --tuner_type full \
    --dataset "$DATASET" "${DATASET_ARGS[@]}" \
    --seq_kd false --lmbda 1 --beta 0.5 --teacher_weights 0.5 0.5 \
    --temperature 0.7 --top_p 0.8 --top_k 20 --repetition_penalty 1.2 --enable_thinking false \
    --use_vllm_teachers true \
    --vllm_teacher_1_url "$T1_URL" --vllm_teacher_2_url "$T2_URL" \
    --vllm_sidecar_1_url "$S1_URL" --vllm_sidecar_2_url "$S2_URL" \
    --torch_dtype bfloat16 --num_train_epochs 1 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 --weight_decay 0.01 --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 16384 --max_completion_length 4096 \
    --attn_impl flash_attn --gradient_checkpointing true \
    --deepspeed zero3 --use_logits_to_keep true \
    --save_strategy steps --save_steps 5 --save_only_model true --logging_steps 1 \
    --output_dir "$OUTPUT" --report_to tensorboard "$@"
