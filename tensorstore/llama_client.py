import torch
import logging
from multiprocessing.managers import BaseManager, DictProxy
import time
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] [LlamaClient] - %(message)s')

MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50001
MANAGER_AUTHKEY = b'temp_authkey'

class TensorManager(BaseManager):
    pass

TensorManager.register('get_tensors', proxytype=DictProxy)

class SharedLlamaModel:
    """
    공유 메모리에서 가중치를 로드하여 Llama 모델 추론을 수행하는 클래스.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model: LlamaForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.manager: BaseManager | None = None
        self.is_loaded = False
        logging.info(f"Initialized SharedLlamaModel for '{model_name}'. Weights not loaded yet.")

    def _connect_to_manager(self) -> bool:
        """TensorManager 서버에 연결합니다."""
        if self.manager and getattr(self.manager, '_state', None) and getattr(self.manager._state, 'value', 0) == 1: # Already connected
             logging.debug("Already connected to manager.")
             return True

        logging.info(f"Attempting to connect to TensorManager server at {MANAGER_HOST}:{MANAGER_PORT}...")
        self.manager = TensorManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=MANAGER_AUTHKEY)
        max_retries = 5
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                self.manager.connect()
                logging.info("Connected to TensorManager server.")
                return True
            except ConnectionRefusedError:
                logging.warning(f"Connection refused (Attempt {attempt + 1}/{max_retries}). Retrying...")
                if attempt == max_retries - 1:
                    logging.error("Max connection attempts reached.")
                    return False
                time.sleep(retry_delay)
            except Exception as e:
                logging.exception(f"Error connecting to manager: {e}")
                return False
        return False # Should not be reached

    def load_weights_from_shared_memory(self) -> bool:
        """공유 메모리에서 모델 가중치를 로드하고 시간을 측정합니다."""
        if self.is_loaded:
            logging.info("Weights are already loaded.")
            return True

        if not self._connect_to_manager():
            return False

        # --- 시간 측정 시작 ---
        init_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats() # 현재 GPU의 통계 리셋
            initial_peak_mem = torch.cuda.max_memory_allocated() # 리셋 직후 (대개 0)
            logging.info(f"Initial peak GPU memory allocated: {initial_peak_mem / (1024**2):.2f} MB")

        try:
            logging.info("Accessing shared tensors via manager...")
            shared_tensor_data_proxy = self.manager.get_tensors()
            if not shared_tensor_data_proxy:
                 logging.error("Failed to get shared tensors or proxy is empty.")
                 return False

            tensor_names = list(shared_tensor_data_proxy.keys())
            logging.info(f"Received access proxy to {len(tensor_names)} shared tensors.")
            if not tensor_names:
                 logging.error("No tensor names found in the shared proxy.")
                 return False

            # 1. 모델 구조 생성 (가중치 없이) - LlamaConfig 사용
            logging.info("Creating model structure from config...")
            config = LlamaConfig.from_pretrained(self.model_name)
            # config에 dtype 설정 시도 (필수는 아닐 수 있음, 모델 생성 후 .to(dtype)도 가능)
            config.torch_dtype = torch.float16

            # 모델 클래스 생성자에 config 전달하여 인스턴스화
            logging.info("Instantiating model from config...")
            self.model = LlamaForCausalLM(config)
            logging.info("Model structure instantiated successfully.")

            # 2. 공유 텐서를 사용하여 state_dict 구성
            logging.info("Building state_dict from shared tensors...")
            state_dict_to_load = {}
            loaded_count = 0
            expected_params = set(n for n, p in self.model.named_parameters())
            shared_params_found = set()

            for name, shared_tensor in shared_tensor_data_proxy.items():
                if name in expected_params:
                    local_shared_tensor = shared_tensor
                    state_dict_to_load[name] = local_shared_tensor
                    shared_params_found.add(name)
                    loaded_count += 1
                    logging.debug(f"  Added shared tensor '{name}' to state_dict.")
                else:
                     logging.warning(f"  Shared tensor '{name}' not found in local model structure.")

            missing_keys = expected_params - shared_params_found
            if missing_keys:
                logging.warning(f"Missing keys in shared data needed by local model: {missing_keys}")

            logging.info(f"Built state_dict with {loaded_count} tensors.")

            # --- lm_head.weight 처리 추가 ---
            if 'lm_head.weight' not in state_dict_to_load and 'model.embed_tokens.weight' in state_dict_to_load:
                logging.info("Manually tying lm_head.weight to model.embed_tokens.weight.")
                state_dict_to_load['lm_head.weight'] = state_dict_to_load['model.embed_tokens.weight']
            # ------------------------------

            # 3. 모델 구조를 CUDA로 이동하고 state_dict 로드
            logging.info("Moving model structure to CUDA...")
            self.model.to('cuda')

            logging.info("Loading state_dict from shared tensors...")
            result = self.model.load_state_dict(state_dict_to_load, strict=False)
            # load_state_dict 후 lm_head 가 올바르게 로드되었는지 확인 (선택적)
            if 'lm_head.weight' in state_dict_to_load:
                if not torch.equal(self.model.lm_head.weight.data, state_dict_to_load['lm_head.weight']): # .data 로 접근해야 할 수도 있음
                    logging.warning("lm_head.weight data might not be correctly loaded/tied.")

            logging.info(f"load_state_dict result - Missing keys: {result.missing_keys}, Unexpected keys: {result.unexpected_keys}")

            # missing_keys 에 lm_head.weight 가 없어야 함 (수동 처리했으므로)
            if 'lm_head.weight' in result.missing_keys:
                 logging.error("lm_head.weight was reported missing despite manual tying attempt!")

            init_end_time = time.time()
            self.is_loaded = True
            self.model.eval()
            init_duration = init_end_time - init_start_time
            logging.info(f"Successfully loaded weights from shared memory into the model in {init_duration:.2f} seconds.")

            return True

        except Exception as e:
            logging.exception(f"Failed to load weights from shared memory: {e}")
            self.model = None
            self.is_loaded = False
            return False

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str | None:
        """주어진 프롬프트로 텍스트를 생성하고 시간을 측정합니다."""
        if not self.is_loaded or self.model is None:
            logging.error("Model weights not loaded. Call load_weights_from_shared_memory() first.")
            return None

        # 토크나이저 로드 (처음 호출 시) - AutoTokenizer 사용
        if self.tokenizer is None:
            logging.info(f"Loading tokenizer for '{self.model_name}'...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logging.info("Tokenizer loaded.")
            except Exception as e:
                logging.exception(f"Failed to load tokenizer: {e}")
                return None

        logging.info(f"Generating text deterministically for prompt: \"{prompt[:50]}...\"")
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')

        # --- 생성 시간 측정 시작 ---
        gen_start_time = time.time()
        try:
            with torch.no_grad():
                # 결정적 생성을 위해 do_sample=False 설정
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                              do_sample=False) # 샘플링 비활성화
            # --- 생성 시간 측정 종료 ---
            gen_end_time = time.time()
            gen_duration = gen_end_time - gen_start_time
            logging.info(f"Text generation completed in {gen_duration:.2f} seconds.")

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.debug(f"Generated output:\n{generated_text}")

            # --- 추론 후 GPU 메모리 측정 ---
            if torch.cuda.is_available():
                 post_inference_peak_mem = torch.cuda.max_memory_allocated()
                 logging.info(f"Peak GPU memory after inference: {post_inference_peak_mem / (1024**2):.2f} MB")

            return generated_text
        except Exception as e:
            logging.exception(f"Error during text generation: {e}")
            return None

# --- Main Execution ---
if __name__ == "__main__":
    server_model_name = "meta-llama/Llama-3.2-1B-Instruct" # Hardcoded for example

    logging.info("--- Llama Client with Shared Weights ---")
    llama_client = SharedLlamaModel(server_model_name)

    # 공유 메모리에서 가중치 로드 시도
    if llama_client.load_weights_from_shared_memory():
        logging.info("--- Performing Sample Inference ---")
        sample_prompt = "The capital of South Korea is"

        # 추론 실행
        generated_output = llama_client.generate(sample_prompt, max_new_tokens=60)

        if generated_output:
            logging.info("--- Inference Result ---")
            print("="*30)
            print(f"Prompt: {sample_prompt}")
            print("-"*30)
            print(f"Generated: {generated_output}")
            print("="*30)
        else:
            logging.error("Inference failed.")
    else:
        logging.error("Failed to load model weights from shared memory. Cannot perform inference.")

    logging.info("--- Llama Client Finished ---") 