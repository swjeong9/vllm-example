from safetensors import safe_open
from multiprocessing.managers import BaseManager, DictProxy
import torch.multiprocessing as mp
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] [Server] - %(message)s')

TMP_SAFETENSOR_PATH = "/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/model.safetensors"

TENSOR_DICT = {}

MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50001
MANAGER_AUTHKEY = b'param_store'

DTYPE = torch.float16

start_layer_idx = 0
end_layer_idx = 16 # 16을 포함하지 않음

def get_tensor_dict():
    return TENSOR_DICT

class TensorManager(BaseManager):
    pass

# 매니저에 공유 데이터 접근 함수 등록 (get_tensors)
# get_tensor 이름 함수 등록하고 DictProxy 사용
TensorManager.register('get_tensor_dict', callable=get_tensor_dict, proxytype=DictProxy)

def main():
    try:
        mp.set_start_method('fork', force=True)
        logging.info("Set multiprocessing start method to 'fork'.")
    except RuntimeError:
        logging.warning("Could not set start method to 'fork'. Using default.")

    logging.info("Loading model directly in the main process...")

    # 모델 로딩 해서 텐서에 저장
    try:
        with safe_open(TMP_SAFETENSOR_PATH, framework="pt", device="cuda") as f:
            for tensor_name in f.keys():
                if tensor_name.startswith("model.layers"):
                    layer_idx = int(tensor_name.split('.')[2])
                    if not (start_layer_idx <= layer_idx < end_layer_idx):
                        continue
                TENSOR_DICT[tensor_name] = f.get_tensor(tensor_name).to(dtype=DTYPE)
                logging.info(f"Loaded {tensor_name} / shape: {TENSOR_DICT[tensor_name].shape}")
    except Exception as e:
        logging.error(f"Model Loading 중 Error 발생")
        raise e
    
    if not TENSOR_DICT:
        raise ValueError("텐서 로딩 실패 : TENSOR_DICT 가 비어있음")
    
    manager = TensorManager(address=(MANAGER_HOST, MANAGER_PORT), authkey=MANAGER_AUTHKEY)

    try:
        logging.info(f"TensorManager 서버 시작 {MANAGER_HOST}:{MANAGER_PORT}...")
        server = manager.get_server()
        logging.info("Manager server running. Waiting for client connections...")
        logging.info("Press Ctrl+C to stop.")
        server.serve_forever() # 블로킹 호출

    except OSError as bind_e:
        logging.error(f"Failed to bind manager server: {bind_e}. Port {MANAGER_PORT} might be in use.")
    except KeyboardInterrupt:
        logging.info("Received Ctrl+C. Shutting down manager server...")
    except Exception as e:
        logging.exception(f"An unexpected error occurred in manager server loop: {e}")
    finally:
        logging.info("Server shutting down.")

if __name__ == "__main__":
    main()