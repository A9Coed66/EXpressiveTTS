import os
import psutil

def set_cpu_affinity(core_ids):
    # return
    """Giới hạn process và các subprocess chỉ chạy trên core 0 và 1"""
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity(core_ids)  # Chỉ dùng core 0 và 1
        print(f"Process {os.getpid()} bị giới hạn trên core: {p.cpu_affinity()}")
    except Exception as e:
        print(f"Không thể thiết lập cpu_affinity: {e}")

