#!/usr/bin/env bash

MODEL="deepseek-v4-flash"
BACKEND="openai"
DATASET="data/locomo10.json"
SAMPLE_IDS="0"
MAX_PER_SAMPLE=""

# attention rerank model (local HF path/name)
ATTENTION_MODEL="/data/hyc/models/Meta-Llama-3.1-8B-Instruct/"
ATTENTION_MAX_LENGTH="4096"

ARGS=(
  --mode attention
  --backend "${BACKEND}"
  --model "${MODEL}"
  --dataset "${DATASET}"
  --retrieve_k 5
  --results_dir results
  --attention_model "${ATTENTION_MODEL}"
  --attention_max_length "${ATTENTION_MAX_LENGTH}"
)

if [[ -n "${SAMPLE_IDS}" ]]; then ARGS+=( --sample_ids "${SAMPLE_IDS}" ); fi
if [[ -n "${MAX_PER_SAMPLE}" ]]; then ARGS+=( --max_questions_per_sample "${MAX_PER_SAMPLE}" ); fi

python test/test_advanced_robust.py "${ARGS[@]}"

