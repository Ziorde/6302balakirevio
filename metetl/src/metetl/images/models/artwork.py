"""
Модуль с абстрактным базовым классом Artwork
"""
from abc import ABC, abstractmethod
from typing import Dict
import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from .metadata import Metadata

logger = logging.getLogger("metetl.images.models")


class Artwork(ABC):
    """
    Абстрактный базовый класс для произведения искусства
    """
    __slots__ = ('_image', '_metadata', '_processed_images')

    def __init__(self, image: NDArray[np.uint8], metadata: Metadata):
        """
        Инициализация произведения искусства

        Args:
            image: Изображение в виде numpy массива
            metadata: Метаданные изображения
        """
        self._image = image.copy()
        self._metadata = metadata
        self._processed_images: Dict[str, NDArray[np.uint8]] = {}

        logger.debug(f"Создан Artwork: {metadata}")

    @property
    def image(self) -> NDArray[np.uint8]:
        """Геттер для изображения"""
        return self._image.copy()

    @property
    def metadata(self) -> Metadata:
        """Геттер для метаданных"""
        return self._metadata

    @property
    def shape(self) -> tuple:
        """Свойство для получения формы изображения"""
        return self._image.shape

    def __str__(self) -> str:
        """Перегрузка метода преобразования в строку"""
        return f"{self._metadata}\nРазмеры: {self.shape}"

    def __add__(self, other: 'Artwork') -> NDArray[np.uint8]:
        """
        Перегрузка оператора сложения - горизонтальное объединение изображений

        Args:
            other: Другое произведение искусства

        Returns:
            NDArray[np.uint8]: Объединённое изображение
        """
        logger.debug(f"Объединение изображений: {self._metadata.object_id} + {other._metadata.object_id}")

        if self.shape[:2] != other.shape[:2]:
            target_height = min(self.shape[0], other.shape[0])
            img1 = self._resize_to_height(target_height)
            img2 = other._resize_to_height(target_height)
        else:
            img1 = self._image
            img2 = other._image

        return np.hstack((img1, img2))

    def _resize_to_height(self, target_height: int) -> NDArray[np.uint8]:
        """
        Изменение размера изображения до заданной высоты с сохранением пропорций

        Args:
            target_height: Целевая высота

        Returns:
            NDArray[np.uint8]: Изменённое изображение
        """
        aspect_ratio = self._image.shape[1] / self._image.shape[0]
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(self._image, (target_width, target_height))

    @abstractmethod
    def rgb_to_grayscale(self) -> NDArray[np.uint8]:
        """Преобразование в полутоновое изображение"""
        pass

    @abstractmethod
    def convolution(self, kernel: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Применение свёртки"""
        pass

    @abstractmethod
    def gaussian_blur(self, kernel_size: int = 5, sigma: float = 1.0) -> NDArray[np.uint8]:
        """Гауссово размытие"""
        pass

    @abstractmethod
    def sobel_edge_detection(self) -> NDArray[np.uint8]:
        """Выделение границ оператором Собеля"""
        pass

    @abstractmethod
    def gamma_correction(self, gamma: float = 0.5) -> NDArray[np.uint8]:
        """Гамма-коррекция"""
        pass

    @abstractmethod
    def histogram_equalization(self) -> NDArray[np.uint8]:
        """Выравнивание гистограммы"""
        pass

    def save_processed(self, name: str, image: NDArray[np.uint8]) -> None:
        """
        Сохранение обработанного изображения

        Args:
            name: Название обработки
            image: Обработанное изображение
        """
        self._processed_images[name] = image.copy()
        logger.debug(f"Сохранен результат обработки '{name}' для {self._metadata.object_id}")