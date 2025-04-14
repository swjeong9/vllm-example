import os
from typing import List, Dict
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.usage.usage_lib import UsageContext
import ray
from torch.distributed import destroy_process_group
from utils import create_placement_group_and_bundle_indices

# 각 노드 IP에 할당할 rank 리스트를 정의합니다.
# 예시: node_rank_mapping = {
#     "172.31.20.243": [0],
#     "172.31.10.100": [1, 2], # 172.31.10.100 노드에는 rank 1과 2 할당
#     "172.31.30.50": [3]    # 172.31.30.50 노드에는 rank 3 할당
# }
node_rank_mapping = {
    "172.31.37.222": [0]
}

async def main():
    model = "meta-llama/Llama-3.2-1B-Instruct"
    task = "generate"
    dtype = "float16"

    engine_args = {
        "model": model,
        "task": task,
        "dtype": dtype,
        "parallel_strategy": [1],
        # "tensor_parallel_size": 1,
        # "pipeline_parallel_size": 1,
    }
    engine_args = AsyncEngineArgs(**engine_args)
    usage_context = UsageContext.ENGINE_CONTEXT

    # 해당 코드에서 config.json 을 받아오며 engine 버전을 어떤 것을 사용할 것인지 정하게 됨.
    # 예를 들어 V1 을 사용하려 했으나 Compute Capacity 가 80아래면 V0 로 Fall back 하게 됨
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    vllm_config.model_config.max_model_len = 4096

    placement_group = create_placement_group_and_bundle_indices(node_rank_mapping)
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
