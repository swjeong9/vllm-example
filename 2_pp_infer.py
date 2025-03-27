import os
from typing import List, Dict
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.usage.usage_lib import UsageContext
import ray
from torch.distributed import destroy_process_group

node_rank_mapping = {
    "172.31.16.230": 0,
    "172.31.9.40": 2,
    "172.31.24.180": 1,
}

def create_placement_group_and_bundle_indices():
    if not ray.is_initialized():
        ray.init(address="auto")

    nodes = ray.nodes()
    ips = [ip for ip, rank in node_rank_mapping.items()]

    placement_group_specs: List[Dict[str, float]] = []
    for ip in ips:
        placement_group_specs.append({
            'GPU': 1,
            f"node:{ip}": 0.001 # 특정 노드를 사용하겠다는 의미
        })
    
    placement_group = ray.util.placement_group(placement_group_specs, strategy="STRICT_SPREAD")
    ray.get(placement_group.ready())

    bundle_to_node = {}
    for bundle_id, bundle in enumerate(placement_group.bundle_specs):
        for resource_key in bundle:
            if resource_key.startswith("node:"):
                node_ip = resource_key[5:] # 'node:172.31.16.230' -> '172.31.16.230'
                bundle_to_node[bundle_id] = node_ip
    
    bundle_indices = []
    for rank in range(len(node_rank_mapping)):
        for bundle_idx, node_ip in bundle_to_node.items():
            if node_rank_mapping[node_ip] == rank:
                bundle_indices.append(str(bundle_idx))
                break
            
    os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(bundle_indices)
    print(f"VLLM_RAY_BUNDLE_INDICES is setted to {os.environ['VLLM_RAY_BUNDLE_INDICES']}")
    print(f"bundle specs : {placement_group.bundle_specs}")

    # 우선 매뉴얼하게 박아보자
    os.environ["VLLM_PP_LAYER_PARTITION"] = "12,14,6"

    return placement_group


async def main():
    model = "meta-llama/Llama-2-7b-chat-hf"
    task = "generate"
    dtype = "float16"

    engine_args = {
        "model": model,
        "task": task,
        "dtype": dtype,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 3,
    }
    engine_args = AsyncEngineArgs(**engine_args)
    usage_context = UsageContext.ENGINE_CONTEXT

    # 해당 코드에서 config.json 을 받아오며 engine 버전을 어떤 것을 사용할 것인지 정하게 됨.
    # 예를 들어 V1 을 사용하려 했으나 Compute Capacity 가 80아래면 V0 로 Fall back 하게 됨
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    vllm_config.model_config.max_model_len = 4096

    placement_group = create_placement_group_and_bundle_indices()
    vllm_config.parallel_config.placement_group = placement_group

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
