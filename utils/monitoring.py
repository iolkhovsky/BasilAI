import os
import psutil


def get_ram_consumption_mb():
    process = psutil.Process(os.getpid())
    consumption_bytes = process.memory_info().rss
    return int(consumption_bytes / 10**6)
