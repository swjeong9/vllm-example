import torch
import logging
from transformers import LlamaForCausalLM, AutoTokenizer
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] [NaiveLlama] - %(message)s')

def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct" # llama_client와 동일한 모델

    logging.info(f"--- Naive Llama Inference (Direct Loading) for {model_name} ---")

    try:
        # --- GPU 메모리 측정 시작점 ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats() # 통계 리셋
            initial_peak_mem = torch.cuda.max_memory_allocated()
            logging.info(f"Initial peak GPU memory allocated: {initial_peak_mem / (1024**2):.2f} MB")

        # 1. 토크나이저 로드
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Tokenizer loaded.")

        # 2. 모델 직접 로드 (from_pretrained 사용) + 시간 측정
        logging.info("Loading model with weights directly using from_pretrained...")
        init_start_time = time.time() # 초기화 시간 측정 시작
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to('cuda:0')
        init_end_time = time.time() # 초기화 시간 측정 종료
        init_duration = init_end_time - init_start_time
        logging.info(f"Model loaded and moved to CUDA in {init_duration:.2f} seconds.")

        model.eval()

        # --- 초기화 후 GPU 메모리 측정 ---
        if torch.cuda.is_available():
             post_init_peak_mem = torch.cuda.max_memory_allocated()
             logging.info(f"Peak GPU memory after initialization: {post_init_peak_mem / (1024**2):.2f} MB")

        # 3. 샘플 추론 수행 + 시간 측정 + 결정적 실행
        logging.info("--- Performing Sample Inference ---")
        sample_prompt = "The capital of South Korea is"
        max_new_tokens=60

        logging.info(f"Generating text deterministically for prompt: \"{sample_prompt[:50]}...\"")
        inputs = tokenizer(sample_prompt, return_tensors="pt").to(model.device)

        # --- 생성 시간 측정 시작 ---
        gen_start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        # --- 생성 시간 측정 종료 ---
        gen_end_time = time.time()
        gen_duration = gen_end_time - gen_start_time
        logging.info(f"Text generation completed in {gen_duration:.2f} seconds.")

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # --- 추론 후 GPU 메모리 측정 ---
        if torch.cuda.is_available():
             post_inference_peak_mem = torch.cuda.max_memory_allocated()
             logging.info(f"Peak GPU memory after inference: {post_inference_peak_mem / (1024**2):.2f} MB")

        logging.info("--- Inference Result ---")
        print("="*30)
        print(f"Prompt: {sample_prompt}")
        print("-"*30)
        print(f"Generated: {generated_text}")
        print("="*30)

    except Exception as e:
        logging.exception(f"An error occurred during naive inference: {e}")

    logging.info("--- Naive Llama Inference Finished ---")


if __name__ == "__main__":
    main() 