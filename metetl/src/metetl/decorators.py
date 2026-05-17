"""
Декораторы для проекта metetl.
"""

import time
import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger("metetl.decorators")


def timer_decorator(func: Callable) -> Callable:
    """
    Декоратор для замера времени выполнения функции.

    Args:
        func: Декорируемая функция

    Returns:
        Callable: Обернутая функция с замером времени
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        logger.debug(f"Начало выполнения {func.__name__}")

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.debug(
                f"Завершение {func.__name__}, "
                f"время выполнения: {execution_time:.4f} сек"
            )

    return wrapper


def async_timer_decorator(func: Callable) -> Callable:
    """
    Декоратор для замера времени выполнения асинхронной функции.

    Args:
        func: Декорируемая асинхронная функция

    Returns:
        Callable: Обернутая асинхронная функция с замером времени
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        logger.debug(f"Начало выполнения асинхронной {func.__name__}")

        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.debug(
                f"Завершение асинхронной {func.__name__}, "
                f"время выполнения: {execution_time:.4f} сек"
            )

    return wrapper