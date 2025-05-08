#!/bin/bash

python benchmark_serving.py --backend vllm \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset-name sharegpt \
    --dataset-path=/home/ubuntu/vllm-example/benchmark/dataset/sharegpt/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
    --save-result --save-detailed \
    --num-prompts 1000 \
    --seed 42 \
    --metric-percentiles "25,50,75,100" \
    --percentile-metrics ttft,tpot,itl,e2el \
    # --request-rate=0.1 \
    # --max-concurrency=4 \
    --result-dir=./results --metadata pipeline-parallel-strategy=1,4 nodes=g4dn.xlarge,g4dn.12xlarge