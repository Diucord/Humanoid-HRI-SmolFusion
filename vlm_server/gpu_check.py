# gpu_check.py
import torch

def check_gpu_available():
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA가 사용 불가능합니다. GPU 환경을 확인해주세요.")
    else:
        print(f"✅ GPU 사용 가능 (장치: {torch.cuda.get_device_name(0)})")
