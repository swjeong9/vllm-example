import time

module_loading_start = time.time()
import os
from typing import List, Dict
vllm_loading_start = time.time()
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
vllm_loading_end = time.time()
print(f"vllm loading time: {vllm_loading_end - vllm_loading_start} seconds")
from vllm.usage.usage_lib import UsageContext
from torch.distributed import destroy_process_group
from utils import create_placement_group_and_bundle_indices
import json
module_loading_end = time.time()
print(f"total module loading time: {module_loading_end - module_loading_start} seconds")
# 각 노드 IP에 할당할 rank 리스트를 정의합니다.
# 예시: node_rank_mapping = {
#     "172.31.20.243": [0],
#     "172.31.10.100": [1, 2], # 172.31.10.100 노드에는 rank 1과 2 할당
#     "172.31.30.50": [3]    # 172.31.30.50 노드에는 rank 3 할당
# }
node_rank_mapping = json.load(open("node_rank_mapping.json"))

async def main():
    model = "meta-llama/Llama-3.2-1B-Instruct"
    task = "generate"
    dtype = "float16"

    engine_args = {
        "model": model,
        "task": task,
        "dtype": dtype,
        "parallel_strategy": [1],
        "enforce_eager": True,
        # "tensor_parallel_size": 1,
        # "pipeline_parallel_size": 1,
    }
    start_time = time.time()
    engine_args = AsyncEngineArgs(**engine_args)
    end_time = time.time()
    print(f"engine_args loading time: {end_time - start_time} seconds")
    usage_context = UsageContext.ENGINE_CONTEXT

    # 해당 코드에서 config.json 을 받아오며 engine 버전을 어떤 것을 사용할 것인지 정하게 됨.
    # 예를 들어 V1 을 사용하려 했으나 Compute Capacity 가 80아래면 V0 로 Fall back 하게 됨
    start_time = time.time()
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    end_time = time.time()
    print(f"vllm_config loading time by create_engine_config(): {end_time - start_time} seconds")
    vllm_config.model_config.max_model_len = 4096

    start_time = time.time()
    placement_group = create_placement_group_and_bundle_indices(node_rank_mapping)
    end_time = time.time()
    print(f"placement_group loading time by create_placement_group_and_bundle_indices(): {end_time - start_time} seconds")
    vllm_config.parallel_config.placement_group = placement_group
    os.environ["VLLM_PP_LAYER_PARTITION"] = "16"

    # 아래 코드에서 tokenizer 와 model 의 safetensor 를 받아오며
    # CUDA graph shape capturing 을 한다.
    engine = AsyncLLMEngine.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_stats=engine_args.disable_log_stats,
            )
    
    prompt = "What is LLM?"
    example_input = {
        "prompt": prompt,
        "stream": False, # assume the non-streaming case
        "temperature": 0.0,
        "request_id": 0,
    }

    # start the generation
    results_generator = engine.generate(
        example_input["prompt"],
        SamplingParams(temperature=example_input["temperature"]),
        example_input["request_id"])

    # get the results
    final_output = None

    # 실행한 메인 스레드에서만 출력하게 설정되어 있음
    async for request_output in results_generator:
        for text_output in request_output.outputs:
            print(f"Prompt: {prompt}")
            print(f"Output: {text_output.text.strip()}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    destroy_process_group()
