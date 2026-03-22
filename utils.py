"""
Утилиты для работы с изображениями и замера времени выполнения
"""

import os
import time
from typing import Any, Callable, Optional, Tuple

import cv2

import numpy as np
from numpy.typing import NDArray

import config


def jpg_to_array(image_path: str) -> Optional[NDArray[np.uint8]]:
    """
    Преобразует JPG изображение в массив numpy

    Args:
        image_path: Путь к JPG файлу

    Returns:
        Optional[NDArray[np.uint8]]: Массив изображения или None в случае ошибки
    """
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл {image_path} не найден")
        return None

    if not image_path.lower().endswith(('.jpg', '.jpeg', '.jpe')):
        print(f"Предупреждение: Файл {image_path} может не быть JPG")

    img = cv2.imread(image_path)

    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return None

    return img


def save_array_to_jpg(
    array: NDArray[np.uint8],
    filename: str,
) -> bool:
    """
    Сохраняет массив обратно в JPG файл

    Args:
        array: Массив изображения
        filename: Имя файла для сохранения

    Returns:
        bool: True если сохранение успешно, False в случае ошибки
    """
    try:
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        output_path = os.path.join(config.SAVE_DIR, filename)
        cv2.imwrite(output_path, array)
        print(f"Результат сохранён в {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении: {e}")
        return False

    return True


def measure_time_and_save(
    processing_func: Callable[..., NDArray[np.uint8]],
    image: NDArray[np.uint8],
    output_filename: str,
    *args: Any,
) -> Tuple[Optional[NDArray[np.uint8]], float]:
    """
    Функция для замера времени выполнения и сохранения результата

    Args:
        processing_func: Функция обработки изображения
        image: Входное изображение в виде массива
        output_filename: Имя файла для сохранения результата
        *args: Дополнительные аргументы для функции обработки

    Returns:
        Tuple[Optional[NDArray[np.uint8]], float]: Кортеж (результат обработки,
                                                   время выполнения в секундах)
    """
    start_time = time.time()

    result = processing_func(image, *args)

    end_time = time.time()
    execution_time = end_time - start_time

    if result is not None:
        save_array_to_jpg(result, output_filename)
    else:
        print("Ошибка: Функция обработки вернула None")

    func_name = processing_func.__name__
    print(
        f"Функция '{func_name}': {execution_time:.4f} сек | "
        f"Сохранено в: {output_filename}"
    )

    return result, execution_time
