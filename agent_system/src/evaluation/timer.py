import time
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

@contextmanager
def timer(task_name: str):
    """Context manager for timing tasks in minutes"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = (time.time() - start_time) / 60  # 초를 분으로 변환
        logger.info(f"{task_name} completed in {elapsed_time:.2f} minutes")