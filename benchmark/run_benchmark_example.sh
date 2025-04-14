#!/bin/bash

python benchmark_serving.py --backend vllm \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dataset-name sharegpt \
    --dataset-path=/home/ubuntu/vllm-example/benchmark/dataset/sharegpt/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
    --save-result --save-detailed \
    --seed 42 \
    --result-dir=./results --metadata pipeline-parallel-strategy=1,4 nodes=g4dn.xlarge,g4dn.12xlarge