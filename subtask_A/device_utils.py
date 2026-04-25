import os
from typing import Literal

def get_device() -> Literal['cuda', 'cpu']:
    device_config = os.getenv('DEVICE', 'auto')
    
    if device_config != 'auto' and device_config in ['cuda', 'cpu']:
        return device_config
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    except (ImportError, OSError):
        # PyTorch not available or DLL error on Windows
        return 'cpu'


def is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, OSError):
        return False


def get_device_info() -> dict:
    device = get_device()
    info = {
        'device': device,
        'cuda_available': is_cuda_available()
    }
    
    if info['cuda_available']:
        try:
            import torch
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    
    return info

if __name__ == '__main__':
    info = get_device_info()
    print("Device Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
