# vllm-example
This project aims to practice vllm framework for developing distributed LLM serving system.
Currently used version of vllm : v0.7.3

### Install build essential & cmake
```bash
sudo apt update
sudo apt install build-essential
sudo apt install cmake
```

### Install cuda toolkit (12.8) & GPU driver (570)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

sudo apt-get install -y nvidia-open-570

echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
sudo reboot
```
```bash
nvcc --version
nvidia-smi --version
```

### Install nccl (https://developer.nvidia.com/nccl/nccl-download)
keyring 은 위에서 이미 받았으므로 과정에서 제외
```bash
sudo apt update
sudo apt install libnccl2=2.26.2-1+cuda12.8 libnccl-dev=2.26.2-1+cuda12.8
```
