import torch
from transformers import LlamaForCausalLM
import time
import socket
import pickle

# Manager를 위한 주소 및 인증키 설정 (예시)
MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50000

def main():
    print("[Server] Loading model directly in the main process...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    try:
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')
        model.eval()
        print(f"[Server] Model '{model_name}' loaded to CUDA.")
    except Exception as e:
        print(f"[Server] Error loading model: {e}")
        return

    ipc_info_dict = {}
    print("[Server] Sharing model parameters via IPC...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.is_cuda:
                try:
                    # 텐서의 저장소를 공유 가능하게 만듦
                    # share_memory_()는 텐서 자체를 변경하고, 공유 정보를 얻기 위해 내부 storage 접근 필요
                    param.share_memory_()
                    # 공유 정보 추출 (주의: 내부 API 사용 가능성)
                    # PyTorch 버전에 따라 storage()._share_cuda_() 또는 유사한 방식 필요
                    # 여기서는 storage().share_cuda_() 가 있다고 가정 (최신 버전 확인 필요)
                    # handle = param.storage().share_cuda_() # 반환값 확인 필요

                    # PyTorch 1.7+ 방식 시도 (내부 API 사용)
                    # storage() -> untyped_storage() 로 변경될 수 있음
                    storage = param.storage()
                    handle = storage._share_cuda_() # 내부 API: (device_ptr, handle, size_bytes)
                    # 실제 핸들 바이트 추출 필요
                    # handle[1] 이 실제 핸들일 수 있음 (버전 확인 필수)
                    ipc_handle_bytes = handle[1] # 핸들 바이트 추출 가정

                    ipc_info_dict[name] = {
                        'handle': ipc_handle_bytes,
                        'size': tuple(param.size()), # size는 튜플로
                        'dtype': param.dtype,
                        'device_index': param.device.index
                    }
                    print(f"[Server]   Shared tensor: {name}")
                except Exception as e:
                    print(f"[Server]   Failed to share tensor {name}: {e}")
            # else: Non-CUDA 파라미터는 무시

    print(f"[Server] Shared {len(ipc_info_dict)} tensors.")

    # 간단한 TCP 소켓 서버 설정
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((MANAGER_HOST, MANAGER_PORT))
            s.listen()
            print(f"[Server] TCP server listening on {MANAGER_HOST}:{MANAGER_PORT}...")

            while True: # 여러 클라이언트 처리 가능 (여기서는 하나만 처리)
                conn, addr = s.accept()
                with conn:
                    print(f"[Server] Connected by {addr}")
                    print("[Server] Sending IPC handle information...")
                    try:
                        # IPC 정보 딕셔너리를 피클링하여 전송
                        serialized_data = pickle.dumps(ipc_info_dict)
                        # 데이터 크기 먼저 전송 (큰 데이터 대비)
                        data_size = len(serialized_data).to_bytes(8, 'big')
                        conn.sendall(data_size)
                        conn.sendall(serialized_data)
                        print("[Server] IPC information sent successfully.")
                        # 전송 후 바로 종료하지 않고, 텐서를 계속 유지해야 함
                        # 클라이언트가 작업을 마칠 때까지 대기하는 로직 추가 가능
                        # 여기서는 무한 루프로 서버 유지 (Ctrl+C로 종료)
                        print("[Server] Keeping tensors alive. Press Ctrl+C to stop.")
                        while True:
                            time.sleep(10)
                    except (socket.error, pickle.PicklingError, Exception) as send_e:
                        print(f"[Server] Error sending data: {send_e}")
                    # break # 하나의 클라이언트만 처리 후 서버 종료 원할 시
        except OSError as bind_e:
            print(f"[Server] Failed to bind socket: {bind_e}. Port might be in use.")
        except KeyboardInterrupt:
            print("[Server] Stopping server...")
        # finally 블록은 with 문이 소켓을 자동으로 닫으므로 필수는 아님
        # finally:
        #      print("[Server] Server shutting down.")

if __name__ == "__main__":
    # 주의: share_memory_ 및 _share_cuda_ API는 버전에 따라 매우 다를 수 있음
    # 실제 사용 시 PyTorch 버전에 맞는 정확한 API 확인 필수
    main()
