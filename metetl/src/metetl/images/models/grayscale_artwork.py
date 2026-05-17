"""
Модуль с конкретной реализацией для чёрно-белого изображения
"""
import logging
import cv2
import numpy as np
from numpy.typing import NDArray

from .artwork import Artwork
from .metadata import Metadata

logger = logging.getLogger("metetl.images.models")


class GrayscaleArtwork(Artwork):
    """
    Класс для чёрно-белого изображения
    """
    __slots__ = ()

    def __init__(self, image: NDArray[np.uint8], metadata: Metadata):
        """
        Инициализация чёрно-белого изображения

        Args:
            image: Чёрно-белое изображение (2D массив) или цветное
            metadata: Метаданные
        """
        if len(image.shape) != 2:
            # Если передано цветное, конвертируем в ч/б
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
            else:
                raise ValueError("GrayscaleArtwork ожидает 2D изображение")
        super().__init__(image, metadata)
        logger.debug(f"Создан GrayscaleArtwork: {metadata.object_id}")

    def rgb_to_grayscale(self) -> NDArray[np.uint8]:
        """Возвращает само изображение (уже ч/б)"""
        return self._image.copy()

    def convolution(self, kernel: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Применение свёртки к ч/б изображению"""
        logger.debug(f"Свёртка ч/б: {self._metadata.object_id}")
        return cv2.filter2D(self._image, -1, kernel).astype(np.uint8)

    def gaussian_blur(self, kernel_size: int = 5, sigma: float = 1.0) -> NDArray[np.uint8]:
        """Гауссово размытие ч/б изображения"""
        logger.debug(f"Гауссово размытие ч/б: {self._metadata.object_id}")
        return cv2.GaussianBlur(self._image, (kernel_size, kernel_size), sigma).astype(np.uint8)

    def sobel_edge_detection(self) -> NDArray[np.uint8]:
        """Выделение границ оператором Собеля"""
        logger.debug(f"Выделение границ ч/б: {self._metadata.object_id}")
        gray = self._image.astype(np.float32)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        if magnitude.max() > 0:
            magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

        return magnitude

    def gamma_correction(self, gamma: float = 0.5) -> NDArray[np.uint8]:
        """Гамма-коррекция ч/б изображения"""
        logger.debug(f"Гамма-коррекция ч/б: {self._metadata.object_id}")
        normalized = self._image.astype(np.float32) / 255.0
        corrected = normalized ** gamma
        return (corrected * 255).astype(np.uint8)

    def histogram_equalization(self) -> NDArray[np.uint8]:
        """Выравнивание гистограммы для ч/б изображения"""
        logger.debug(f"Выравнивание гистограммы ч/б: {self._metadata.object_id}")
        return cv2.equalizeHist(self._image).astype(np.uint8)