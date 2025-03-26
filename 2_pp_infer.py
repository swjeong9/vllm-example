from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.usage.usage_lib import UsageContext

from torch.distributed import destroy_process_group


async def main():
    model = "meta-llama/Llama-2-7b-chat-hf"
    task = "generate"
    dtype = "float16"

    engine_args = {
        "model": model,
        "task": task,
        "dtype": dtype,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 2,
    }
    engine_args = AsyncEngineArgs(**engine_args)
    usage_context = UsageContext.ENGINE_CONTEXT

    # 해당 코드에서 config.json 을 받아오며 engine 버전을 어떤 것을 사용할 것인지 정하게 됨.
    # 예를 들어 V1 을 사용하려 했으나 Compute Capacity 가 80아래면 V0 로 Fall back 하게 됨
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    vllm_config.model_config.max_model_len = 4096

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
    async for request_output in results_generator:
        for text_output in request_output.outputs:
            print(f"Prompt: {prompt}")
            print(f"Output: {text_output.text.strip()}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    destroy_process_group()
