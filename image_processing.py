"""
Модуль с функциями обработки изображений
"""

import config

import cv2

import numpy as np

from numpy.typing import NDArray

def rgb_to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Преобразование RGB изображения в полутоновое
    Используем стандартную формулу: Y = 0.2989 * R + 0.5870 * G + 0.1140 * B

    Args:
        image: Входное RGB изображение в виде массива numpy

    Returns:
        NDArray[np.uint8]: Полутоновое изображение
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def convolution(
    image: NDArray[np.uint8],
    kernel: NDArray[np.float32],
) -> NDArray[np.uint8]:
    """
    Свертка изображения с заданным ядром

    Args:
        image: Входное изображение в виде массива numpy
        kernel: Ядро свертки

    Returns:
        NDArray[np.uint8]: Изображение после свертки
    """
    return cv2.filter2D(image, -1, kernel)


def gaussian_blur(
    image: NDArray[np.uint8],
    kernel_size: int = config.GAUSSIAN_KERNEL_SIZE,
    sigma: float = config.GAUSSIAN_SIGMA,
) -> NDArray[np.uint8]:
    """
    Гауссово размытие

    Args:
        image: Входное изображение в виде массива numpy
        kernel_size: Размер ядра Гаусса (должен быть нечетным)
        sigma: Стандартное отклонение Гауссова ядра

    Returns:
        NDArray[np.uint8]: Размытое изображение
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def sobel_edge_detection(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Выделение границ с помощью оператора Собеля

    Args:
        image: Входное RGB изображение в виде массива numpy

    Returns:
        NDArray[np.uint8]: Изображение с выделенными границами
    """
    # Преобразование в градации серого
    gray = rgb_to_grayscale(image)

    # Вычисляем градиенты
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0)

    # Вычисляем магнитуду
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Нормализуем и преобразуем в uint8
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    return magnitude


def gamma_correction(
    image: NDArray[np.uint8],
    gamma: float = config.GAMMA_VALUE,
) -> NDArray[np.uint8]:
    """
    Гамма-коррекция

    Args:
        image: Входное изображение в виде массива numpy
        gamma: Параметр гамма-коррекции

    Returns:
        NDArray[np.uint8]: Изображение после гамма-коррекции
    """
    # Нормализуем значения в диапазон [0, 1] и применяем гамма-коррекцию
    normalized = image.astype(np.float32) / 255.0
    corrected = normalized ** (1.0 / gamma)

    # Возвращаем в диапазон [0, 255]
    return (corrected * 255).astype(np.uint8)


def histogram_equalization(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Выравнивание гистограммы для цветного RGB изображения

    Args:
        image: Входное RGB изображение в виде массива numpy

    Returns:
        NDArray[np.uint8]: Изображение после выравнивания гистограммы
    """
    # Конвертируем RGB в LAB цветовое пространство
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Разделяем каналы
    lightness, channel_a, channel_b = cv2.split(lab)

    # Выравниваем гистограмму канала яркости
    lightness_eq = cv2.equalizeHist(lightness)

    # Объединяем каналы обратно
    lab_eq = cv2.merge([lightness_eq, channel_a, channel_b])

    # Конвертируем обратно LAB в RGB
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
