from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.usage.usage_lib import UsageContext


async def main():
    model = "meta-llama/Meta-Llama-3-8B"
    task = "generate"
    dtype = "float16"

    engine_args = {
        "model": model,
        "task": task,
        "dtype": dtype,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
    }
    engine_args = AsyncEngineArgs(**engine_args)
    usage_context = UsageContext.ENGINE_CONTEXT
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    vllm_config.model_config.max_model_len = 4096
    engine = AsyncLLMEngine.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_stats=engine_args.disable_log_stats,
            )

    example_input = {
        "prompt": "What is LLM?",
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
            print(f"Output: {text_output.text}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
