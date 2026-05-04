import time
import logging
from functools import wraps
from typing import Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name: str) -> logging.Logger:
    """Создание логгера"""
    return logging.getLogger(name)

def timer_decorator(func: Callable) -> Callable:
    """Декоратор для замера времени выполнения"""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[ТАЙМЕР] {func.__name__}: {execution_time:.4f} сек")
        return result
    return wrapper