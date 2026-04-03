"""
Модуль с декоратором для замера времени выполнения
"""
import time
from functools import wraps
from typing import Any, Callable


def timer_decorator(func: Callable) -> Callable:
    """
    Декоратор для замера времени выполнения функции

    Args:
        func: Декорируемая функция

    Returns:
        Callable: Обёрнутая функция с замером времени
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Функция {func.__name__}: {execution_time:.4f} сек")
        return result

    return wrapper