import argparse
import time
import sys
import os
import json
import subprocess
from typing import Optional, Dict, List, Tuple
from vllm.logger import init_logger

logger = init_logger(__name__)

# 현재 파일의 절대 경로를 가져옵니다.
current_file_path = os.path.abspath(__file__)
# 부모 디렉터리 (tensorstore 폴더의 부모)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
# sys.path에 부모 디렉터리를 추가하여 utils.py를 찾을 수 있도록 합니다.
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import run_server_remote

PYTHON = "/home/ubuntu/.conda/envs/vllm/bin/python"
TENSOR_SERVER_FILE = "/home/ubuntu/vllm-example/tensorstore/mt_tensor_store_server.py"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--node-rank-mapping-path", type=str, required=True)
    parser.add_argument("--pp-partition", type=str, required=True)
    args = parser.parse_args()

    node_rank_mapping = json.load(open(args.node_rank_mapping_path))
    # pp-partition 은 , 로 구분된 int 리스트임.
    pp_partition = [int(x) for x in args.pp_partition.split(",")]

    # json 의 key 수 (node 수) 와 pp-partition 의 length 가 같아야 함.
    assert len(node_rank_mapping) == len(pp_partition), "node 수와 pp-partition 의 length 가 같아야 함."

    layer_idx = 0
    managed_tensor_store_servers: Dict[str, List[Optional[subprocess.Popen]]] = {}
    for node_ip, rank_list in node_rank_mapping.items():
        managed_tensor_store_servers[node_ip] = []
        tensor_parallel_size = len(rank_list)
        for local_rank in range(tensor_parallel_size):
            base_server_command = (
                f"{PYTHON} "
                f"{TENSOR_SERVER_FILE} "
                f"--model-name {args.model_name} "
                f"--tensor-parallel-size {tensor_parallel_size} "
                f"--local-rank {local_rank} "
                f"--start-layer-id {layer_idx} "
                f"--end-layer-id {layer_idx + pp_partition[local_rank]}"
            )

            hostname = node_ip
            username = "ubuntu"
            log_file_path = os.path.join(current_file_path, f"{node_ip}_{local_rank}.log")

            # utils.py의 run_server_remote를 호출한다고 가정 (이전 command.py의 내용과 유사)
            success, process = run_server_remote(hostname, username, base_server_command, log_file_path)
            if not success:
                logger.error(f"[{hostname}] 원격 서버 실행 실패: {process}")
                sys.exit(1)
            else:
                logger.info(f"[{hostname}] 원격 서버 실행 성공: {process}")

            managed_tensor_store_servers[node_ip].append(process)
        layer_idx += pp_partition[local_rank]

    try:
        while True:
            if not managed_tensor_store_servers:
                logger.info("관리할 원격 프로세스가 없습니다. 5초 후 재확인...")
                time.sleep(5)
                if not managed_tensor_store_servers: # 다시 한번 확인 후 종료 결정
                    logger.info("관리할 프로세스가 없어 루프를 종료합니다.")
                    break

            for hostname, proc_obj_list in managed_tensor_store_servers.items(): # dict 변경 중 순회를 위해 list 사용
                for proc_obj in proc_obj_list:
                    return_code = proc_obj.poll()

                    if return_code is None:
                        logger.info(f"[{hostname}] ({time.strftime('%H:%M:%S')}) 원격 서버 실행 중 (SSH PID: {proc_obj.pid}).")
                        # 여기에 원격 로그 파일 확인 로직 추가 가능
                        # 예: check_remote_log(hostname, username, log_file_path_for_this_host)
                    else:
                        logger.error(f"[{hostname}] ({time.strftime('%H:%M:%S')}) 원격 서버가 종료되었거나 연결이 끊어졌습니다! 반환 코드: {return_code}")
                        try:
                            stdout_data, stderr_data = proc_obj.communicate(timeout=1) # 이미 종료되었으므로 짧은 타임아웃
                            if stderr_data:
                                logger.error(f"[{hostname}] SSH STDERR: {stderr_data.decode().strip()}")
                            if stdout_data: # 거의 없을 것으로 예상
                                logger.info(f"[{hostname}] SSH STDOUT: {stdout_data.decode().strip()}")
                        except Exception as e:
                            logger.error(f"[{hostname}] 종료된 SSH 프로세스로부터 stderr 읽기 실패: {e}")
                    
                    time.sleep(1)

                time.sleep(15) # 15초 간격으로 각 노드 프로세스 상태 확인

    except KeyboardInterrupt:
        logger.info("\nCtrl+C 입력 감지. 실행 중인 모든 원격 프로세스를 종료 시도합니다...")
        
        # 종료 대상 프로세스 목록 수집 (호스트명, Popen 객체)
        processes_to_terminate: List[Tuple[str, subprocess.Popen]] = []
        for hostname, proc_obj_list in managed_tensor_store_servers.items():
            for proc_obj in proc_obj_list:
                if isinstance(proc_obj, subprocess.Popen) and proc_obj.poll() is None:
                    processes_to_terminate.append((hostname, proc_obj))

        if not processes_to_terminate:
            logger.info("종료할 활성 원격 프로세스가 없습니다.")
        else:
            # 1단계: 모든 활성 프로세스에 SIGTERM 보내기
            logger.info(f"{len(processes_to_terminate)}개의 활성 원격 프로세스에 SIGTERM 신호를 보냅니다...")
            for hostname, proc_obj in processes_to_terminate:
                logger.info(f"[{hostname}] 원격 프로세스(SSH PID: {proc_obj.pid})에 SIGTERM 전송.")
                try:
                    proc_obj.terminate()
                except Exception as e:
                    logger.error(f"[{hostname}] PID {proc_obj.pid}에 SIGTERM 전송 중 오류: {e}")

            # 종료될 시간 약간 대기
            time.sleep(5)

            # 2단계: 각 프로세스가 종료될 때까지 대기 (또는 타임아웃 후 SIGKILL)
            logger.info("각 원격 프로세스의 종료를 확인 및 대기합니다...")
            for hostname, proc_obj in processes_to_terminate:
                if proc_obj.poll() is None: # SIGTERM 후에도 아직 실행 중이라면
                    logger.info(f"[{hostname}] 원격 프로세스(SSH PID: {proc_obj.pid}) 종료 대기 중 (최대 10초)...")
                    try:
                        proc_obj.wait(timeout=10) # 종료 대기
                        logger.info(f"[{hostname}] 원격 프로세스(SSH PID: {proc_obj.pid})가 SIGTERM에 의해 정상 종료됨 (코드: {proc_obj.returncode}).")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"[{hostname}] 원격 프로세스(SSH PID: {proc_obj.pid})가 SIGTERM에 응답하지 않아 강제 종료(SIGKILL)합니다.")
                        try:
                            proc_obj.kill() # 강제 종료
                            proc_obj.wait(timeout=5) # SIGKILL 후 OS 정리를 위해 짧게 대기
                            logger.info(f"[{hostname}] 원격 프로세스(SSH PID: {proc_obj.pid})가 SIGKILL에 의해 강제 종료됨 (코드: {proc_obj.returncode}).")
                        except subprocess.TimeoutExpired:
                            logger.error(f"[{hostname}] 원격 프로세스(SSH PID: {proc_obj.pid}) 강제 종료 후에도 응답이 없습니다.")
                        except Exception as e_kill:
                            logger.error(f"[{hostname}] PID {proc_obj.pid} 강제 종료 중 오류: {e_kill}")
                    except Exception as e_wait:
                        logger.error(f"[{hostname}] PID {proc_obj.pid} 종료 대기 중 알 수 없는 오류: {e_wait}. 현재 상태: {proc_obj.poll()}")
                        # 위에서 오류 발생 시, 만약 아직도 실행 중이면 최후의 수단으로 kill 시도
                        if proc_obj.poll() is None:
                            logger.warning(f"[{hostname}] 알 수 없는 오류 후 강제 종료(SIGKILL) 재시도 (PID: {proc_obj.pid}).")
                            try:
                                proc_obj.kill()
                                proc_obj.wait(timeout=5)
                            except Exception as e_kill_final:
                                logger.error(f"[{hostname}] PID {proc_obj.pid} 최종 강제 종료 중 오류: {e_kill_final}")
                else: # SIGTERM 전송 후 또는 그 사이에 이미 종료된 경우
                    logger.info(f"[{hostname}] 원격 프로세스(SSH PID: {proc_obj.pid})가 이미 종료되어 있습니다 (코드: {proc_obj.returncode}).")
        
        logger.info("모든 원격 프로세스에 대한 종료 처리 시도 완료.")

    finally:
        logger.info("스크립트 종료.")