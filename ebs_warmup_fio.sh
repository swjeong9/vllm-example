#!/bin/bash

# ==================================================
# EBS 워밍업 및 시간 측정 스크립트
# ==================================================

# 스크립트 설정
set -e # 오류 발생 시 즉시 종료

# --- 설정 변수 ---

# 워밍업 대상: 'DEVICE', 'FILE', 'FOLDER' 중 선택
# DEVICE: 지정된 EBS 볼륨 전체를 워밍업 (fio 사용)
# FILE: 지정된 특정 파일만 워밍업 (fio 사용)
# FOLDER: 지정된 폴더 내 모든 파일을 재귀적으로 워밍업 (find + cat 사용)
WARMUP_TARGET_TYPE="FOLDER" # 또는 "DEVICE", "FILE"

# 대상 장치명 (WARMUP_TARGET_TYPE 이 'DEVICE'일 경우)
# 중요: 'lsblk' 명령 등으로 정확한 장치명을 확인하세요! (예: /dev/nvme1n1, /dev/xvdf)
TARGET_DEVICE="/dev/nvme1n1"

# 대상 파일 경로 (WARMUP_TARGET_TYPE 이 'FILE'일 경우)
# 중요: 워밍업할 대용량 파일의 실제 경로를 지정하세요!
TARGET_FILE="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062/model-00002-of-00002.safetensors"

# 대상 폴더 경로 (WARMUP_TARGET_TYPE 이 'FOLDER'일 경우)
TARGET_FOLDER="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct"

# fio 에서 사용할 블록 크기 (FILE/DEVICE 타입용)
BLOCK_SIZE="1M"

# fio에서 사용할 병렬 작업 수 (CPU 코어 수만큼 사용하려면 $(nproc) 사용)
FIO_NUM_JOBS=$(nproc)

# --- 변수 초기화 ---
elapsed_time_fio=0
elapsed_time_folder=0

echo "=================================================="
echo "EBS 워밍업 스크립트 시작"
echo "=================================================="
echo "대상 유형: $WARMUP_TARGET_TYPE"

# --- 워밍업 실행 ---

if [ "$WARMUP_TARGET_TYPE" == "DEVICE" ] || [ "$WARMUP_TARGET_TYPE" == "FILE" ]; then
    # DEVICE 또는 FILE 타입 공통 처리 (fio 사용)

    # 대상 경로 및 확인 옵션 설정
    if [ "$WARMUP_TARGET_TYPE" == "DEVICE" ]; then
        TARGET=$TARGET_DEVICE
        CHECK_OPT="-b" # 블록 장치 확인
        TARGET_DESC="장치"
        FIO_JOB_NAME="ebs-warmup-device"
    else # FILE 타입
        TARGET=$TARGET_FILE
        CHECK_OPT="-f" # 일반 파일 확인
        TARGET_DESC="파일"
        FIO_JOB_NAME="ebs-warmup-file"
    fi

    # 대상 존재 및 타입 확인
    if [ ! $CHECK_OPT "$TARGET" ]; then
        echo "오류: 지정된 $TARGET_DESC '$TARGET'를 찾을 수 없거나 올바른 타입이 아닙니다."
        exit 1
    fi
    echo "대상 $TARGET_DESC: $TARGET"

    # fio 설치 확인
    if ! command -v fio &> /dev/null; then
        echo "오류: fio가 설치되어 있지 않습니다. (DEVICE/FILE 타입에 필요)"
        echo "설치하려면 다음 명령을 실행하세요:"
        echo "  sudo yum install fio -y  (Amazon Linux, CentOS, RHEL)"
        echo "  sudo apt-get update && sudo apt-get install fio -y (Ubuntu, Debian)"
        exit 1
    fi
    echo "사용될 fio 작업 수: $FIO_NUM_JOBS"
    echo "사용될 블록 크기 (fio): $BLOCK_SIZE"
    echo "--------------------------------------------------"
    echo "fio를 사용하여 $TARGET_DESC 워밍업을 시작합니다..."

    # fio 명령어 구성 (공통)
    FIO_CMD="sudo fio --name=$FIO_JOB_NAME --filename=$TARGET --rw=read --bs=$BLOCK_SIZE --direct=1 --numjobs=$FIO_NUM_JOBS --iodepth=32 --ioengine=libaio --group_reporting"
    echo "명령어: $FIO_CMD"

    start_time_fio=$(date +%s)
    eval $FIO_CMD # fio 실행
    end_time_fio=$(date +%s)
    elapsed_time_fio=$((end_time_fio - start_time_fio))
    echo "fio $TARGET_DESC 워밍업 완료!"

elif [ "$WARMUP_TARGET_TYPE" == "FOLDER" ]; then
    # FOLDER 타입 처리 (find + dd 사용, 파일명 및 진행 상황 표시, 직접 I/O)
    TARGET=$TARGET_FOLDER
    if [ ! -d "$TARGET" ]; then
        echo "오류: 지정된 폴더 '$TARGET'를 찾을 수 없습니다."
        exit 1
    fi
    echo "대상 폴더: $TARGET"
    echo "폴더 내 모든 파일을 재귀적으로 읽어 워밍업을 시작합니다 (find + dd, 파일명 및 진행 상황 표시, 직접 I/O)..."

    start_time_folder=$(date +%s)
    # find 와 dd 를 이용한 워밍업 (각 파일마다 파일명 출력 후 dd 실행, 진행 상황 표시, 직접 I/O)
    # status=progress : dd 실행 중 진행 상황 출력
    # iflag=direct : OS 페이지 캐시를 우회하여 직접 읽기 시도
    # bs=1M : 적절한 블록 크기 설정
    # sh -c '...' : 각 파일에 대해 여러 명령어 실행 (파일명 출력 + dd)
    # $1 : sh -c 에 전달된 파일명 인수
    # \; : 찾은 파일 각각에 대해 sh -c 명령어 실행
    find "$TARGET" -type f -exec sh -c 'echo "--- 처리 중: $1 ---"; dd if="$1" of=/dev/null bs=1M status=progress iflag=direct' sh {} \;
    end_time_folder=$(date +%s)
    elapsed_time_folder=$((end_time_folder - start_time_folder))
    echo # dd 진행 상황 출력 후 줄바꿈 추가
    echo "폴더 워밍업 완료!"

else
    # 잘못된 타입 처리
    echo "오류: WARMUP_TARGET_TYPE은 'DEVICE', 'FILE' 또는 'FOLDER' 이어야 합니다."
    exit 1
fi

echo "--------------------------------------------------"
echo "EBS 워밍업 스크립트 종료"
echo "=================================================="

# 결과 요약
echo "워밍업 소요 시간 요약:"
if [ "$WARMUP_TARGET_TYPE" == "DEVICE" ] || [ "$WARMUP_TARGET_TYPE" == "FILE" ]; then
    echo " - fio (타입: $WARMUP_TARGET_TYPE, jobs=$FIO_NUM_JOBS): ${elapsed_time_fio} 초"
elif [ "$WARMUP_TARGET_TYPE" == "FOLDER" ]; then
    echo " - find + dd (타입: FOLDER, 파일명 표시): ${elapsed_time_folder} 초"
fi