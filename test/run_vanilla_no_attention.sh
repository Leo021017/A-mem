#!/usr/bin/env bash

MODEL="deepseek-v4-flash"
BACKEND="openai"
DATASET="data/locomo10.json"
SAMPLE_IDS="0"
MAX_PER_SAMPLE=""

ARGS=(
  --mode vanilla
  --backend "${BACKEND}"
  --model "${MODEL}"
  --dataset "${DATASET}"
  --retrieve_k 15
  --results_dir results
)

if [[ -n "${SAMPLE_IDS}" ]]; then ARGS+=( --sample_ids "${SAMPLE_IDS}" ); fi
if [[ -n "${MAX_PER_SAMPLE}" ]]; then ARGS+=( --max_questions_per_sample "${MAX_PER_SAMPLE}" ); fi

python test/test_advanced_robust.py "${ARGS[@]}"

