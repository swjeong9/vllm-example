import torch
import torch.multiprocessing as mp
# from multiprocessing.managers import SyncManager # 제거
from multiprocessing.managers import BaseManager # BaseManager 임포트
from queue import Empty
import socket
import pickle

# Manager 주소 및 인증키 (서버와 동일하게 설정)
MANAGER_HOST = '127.0.0.1'
MANAGER_PORT = 50000
# MANAGER_AUTHKEY 제거

# 커스텀 Manager 클래스 제거
# class TorchQueueManager(BaseManager):
#     pass

def open_ipc_tensor(name, ipc_info):
    """IPC 핸들 정보로부터 텐서 뷰를 재구성합니다."""
    try:
        handle_bytes = ipc_info['handle']
        size = ipc_info['size']
        dtype = ipc_info['dtype']
        device_index = ipc_info['device_index']
        device = f"cuda:{device_index}"

        # PyTorch 버전에 따라 필요한 정확한 방법 사용 (매우 중요)

        # 필요한 총 바이트 계산
        element_size = torch.empty((), dtype=dtype).element_size()
        num_elements = 1
        for s in size:
            num_elements *= s
        total_bytes = num_elements * element_size

        # *** 가장 유력한 방법 (내부 API 사용) ***
        # UntypedStorage._new_shared_cuda(ipc_handle, size) 를 사용한다고 가정
        # size 는 byte 단위 크기여야 함
        # PyTorch 2.0+ 에서는 torch.UntypedStorage.from_ipc_handle(handle, size=total_bytes, device=device)
        # 또는 torch.from_file(handle_path, shared=True, size=total_bytes).to(dtype=dtype, device=device) 같은 방식 고려
        # 버전에 맞는 API 확인이 중요! 여기서는 _new_shared_cuda 가정
        storage = torch.UntypedStorage._new_shared_cuda(handle_bytes, total_bytes)
        tensor = torch.empty(size, dtype=dtype, device=device) # 빈 텐서 생성
        tensor.set_(storage) # 공유 메모리를 사용하도록 설정

        print(f"  [Client] Successfully opened tensor: {name}")
        return tensor

    except Exception as e:
        print(f"  [Client] Failed to open tensor {name} from IPC handle: {e}")
        print(f"    Handle info: size={size}, dtype={dtype}, device={device}")
        return None

def main():
    shared_tensors = {}
    print(f"[Client] Connecting to server at {MANAGER_HOST}:{MANAGER_PORT}...")

    try:
        # 서버에 소켓 연결
        with socket.create_connection((MANAGER_HOST, MANAGER_PORT), timeout=10) as s:
            print("[Client] Connected to server.")

            # 데이터 크기 수신
            data_size_bytes = s.recv(8)
            if not data_size_bytes:
                print("[Client] Failed to receive data size from server.")
                return
            data_size = int.from_bytes(data_size_bytes, 'big')
            print(f"[Client] Receiving {data_size} bytes of IPC handle data...")

            # 실제 데이터 수신 (피클된 딕셔너리)
            serialized_data = b""
            received_size = 0
            while received_size < data_size:
                chunk = s.recv(min(4096, data_size - received_size))
                if not chunk:
                    raise ConnectionError("Connection broken while receiving data.")
                serialized_data += chunk
                received_size += len(chunk)

            print("[Client] IPC handle data received.")
            ipc_info_dict = pickle.loads(serialized_data)
            print(f"[Client] Received info for {len(ipc_info_dict)} tensors.")

            print("[Client] Attempting to open shared tensors from IPC handles...")
            for name, ipc_info in ipc_info_dict.items():
                tensor = open_ipc_tensor(name, ipc_info)
                if tensor is not None:
                    shared_tensors[name] = tensor
                    # 간단 테스트 (첫 텐서만)
                    if len(shared_tensors) == 1:
                        try:
                            print(f"    Testing first tensor '{name}'. Norm: {tensor.norm()}")
                        except Exception as test_e:
                            print(f"    Failed to compute norm for tensor '{name}': {test_e}")

            print(f"[Client] Finished opening tensors. {len(shared_tensors)} tensors accessible.")

    except socket.timeout:
        print("[Client] Connection timed out.")
    except ConnectionRefusedError:
        print(f"[Client] Connection refused. Is the server running at {MANAGER_HOST}:{MANAGER_PORT}?")
    except Exception as e:
        print(f"[Client] An error occurred: {e}")

    print("\n[Client] Finished.")
    if shared_tensors:
        print("[Client] Accessible shared tensors:")
        tensor_names = list(shared_tensors.keys())
        if tensor_names:
             print(f"  - First tensor: {tensor_names[0]} (Device: {shared_tensors[tensor_names[0]].device})")
             print(f"  - Last tensor: {tensor_names[-1]}")
             print(f"  Total {len(tensor_names)} tensors accessible.")
    else:
        print("[Client] No tensors were successfully opened.")

if __name__ == "__main__":
    main() 