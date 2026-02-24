"""
Модуль с собственными функциями обработки изображений
"""

import config

import cv2

import numpy as np
from numpy.typing import NDArray


def rgb_to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    """
    Преобразование RGB изображения в полутоновое
    Используем стандартную формулу: Y = 0,2989 * R + 0,5870 * G + 0,1140 * B

    Args:
        image: Входное RGB изображение в виде массива numpy

    Returns:
        NDArray[np.float64]: Полутоновое изображение
    """
    # Отделяем цветовые каналы
    red_channel = image[:, :, 0].astype(np.float64)
    green_channel = image[:, :, 1].astype(np.float64)
    blue_channel = image[:, :, 2].astype(np.float64)

    # Применяем формулу яркости
    gray = (config.GRAYSCALE_WEIGHTS['red'] * red_channel +
            config.GRAYSCALE_WEIGHTS['green'] * green_channel +
            config.GRAYSCALE_WEIGHTS['blue'] * blue_channel)
    return gray


def convolution(
    image: NDArray[np.uint8],
    kernel: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Свертка изображения с заданным ядром

    Args:
        image: Входное изображение в виде массива numpy
        kernel: Ядро свертки

    Returns:
        NDArray[np.float32]: Изображение после свертки
    """
    # Проверяем размерность изображения
    if len(image.shape) == 3:
        # Для RGB изображения применяем свертку к каждому каналу отдельно
        result = np.zeros_like(image, dtype=np.float32)
        for channel in range(image.shape[2]):
            result[:, :, channel] = convolution_2d(
                image[:, :, channel].astype(np.float32),
                kernel,
            )

        return result
    else:
        # Для grayscale изображения
        return convolution_2d(image.astype(np.float32), kernel)


def convolution_2d(
    image: NDArray[np.float32],
    kernel: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Свертка 2D изображения с заданным ядром

    Args:
        image: Входное 2D изображение в виде массива numpy
        kernel: Ядро свертки

    Returns:
        NDArray[np.float32]: Изображение после свертки
    """
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Вычисляем отступы для сохранения размера изображения
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)))
    result = np.zeros_like(image, dtype=np.float32)

    flipped_kernel = np.flip(kernel)
    for i in range(height):
        for j in range(width):
            # Извлекаем область для свертки
            region = padded[i:i + kernel_height, j:j + kernel_width]
            # Вычисляем свертку
            result[i, j] = np.sum(region * flipped_kernel)

    return result


def gaussian_blur(
    image: NDArray[np.uint8],
    kernel_size: int = config.GAUSSIAN_KERNEL_SIZE,
    sigma: float = config.GAUSSIAN_SIGMA,
) -> NDArray[np.float32]:
    """
    Гауссово размытие

    Args:
        image: Входное изображение в виде массива numpy
        kernel_size: Размер ядра Гаусса (должен быть нечетным)
        sigma: Стандартное отклонение Гауссова ядра

    Returns:
        NDArray[np.float32]: Размытое изображение
    """
    # Создаем 1D ядро Гаусса
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = kernel_1d / np.sum(kernel_1d)

    # Создаем 2D ядро как произведение двух 1D ядер
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    # Применяем свертку
    return convolution(image, kernel_2d)


def sobel_edge_detection(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Выделение границ с помощью оператора Собеля

    Args:
        image: Входное RGB изображение в виде массива numpy

    Returns:
        NDArray[np.uint8]: Изображение с выделенными границами
    """
    gray = rgb_to_grayscale(image).astype(np.float32)

    # Применяем свертку с ядрами Собеля
    grad_x = convolution(gray, config.SOBEL_X)
    grad_y = convolution(gray, config.SOBEL_Y)

    # Вычисляем магнитуду градиента
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Нормализуем до 0-255
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

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


def histogram_equalization(
    image: NDArray[np.uint8],
    bins: int = 256,
) -> NDArray[np.uint8]:
    """
    Выравнивание гистограммы для RGB изображения (по яркостному каналу LAB)

    Args:
        image: Входное RGB изображение в виде массива numpy
        bins: Количество бинов гистограммы

    Returns:
        NDArray[np.uint8]: Изображение после выравнивания гистограммы
    """
    # Конвертируем RGB в LAB цветовое пространство
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Разделяем каналы
    lightness, channel_a, channel_b = cv2.split(lab)

    # Выравниваем гистограмму канала яркости
    # Получаем размеры канала яркости
    height, width = lightness.shape
    total_pixels = height * width

    # Вычисление гистограммы для канала яркости
    histogram = [0] * bins
    for i in range(height):
        for j in range(width):
            value = int(lightness[i, j])
            if 0 <= value < bins:
                histogram[value] += 1

    # Вычисление функции распределения CDF
    cdf = [0] * bins
    cdf[0] = histogram[0] / total_pixels

    for i in range(1, bins):
        cdf[i] = cdf[i - 1] + histogram[i] / total_pixels

    # Создание lookup table для выравнивания
    lookup_table = [int(255 * cdf[i]) for i in range(bins)]

    # Применение выравнивания к каналу яркости
    lightness_eq = np.zeros_like(lightness)
    for i in range(height):
        for j in range(width):
            lightness_eq[i, j] = lookup_table[int(lightness[i, j])]

    # Объединяем каналы обратно
    lab_eq = cv2.merge([lightness_eq, channel_a, channel_b])

    # Конвертируем обратно LAB в RGB
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
