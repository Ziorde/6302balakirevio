"""
Модуль с конкретной реализацией для цветного изображения
"""
import cv2
import numpy as np
from numpy.typing import NDArray

from .artwork import Artwork
from .metadata import Metadata

class ColorArtwork(Artwork):
    """
    Класс для цветного изображения
    """
    __slots__ = ()

    def __init__(self, image: NDArray[np.uint8], metadata: Metadata):
        """
        Инициализация цветного изображения

        Args:
            image: Цветное изображение в формате RGB
            metadata: Метаданные
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("ColorArtwork ожидает RGB изображение")
        super().__init__(image, metadata)

    def rgb_to_grayscale(self) -> NDArray[np.uint8]:
        """Преобразование RGB изображения в полутоновое"""
        return cv2.cvtColor(self._image, cv2.COLOR_RGB2GRAY).astype(np.uint8)

    def convolution(self, kernel: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Применение свёртки к цветному изображению"""
        return cv2.filter2D(self._image, -1, kernel).astype(np.uint8)

    def gaussian_blur(self, kernel_size: int = 5, sigma: float = 1.0) -> NDArray[np.uint8]:
        """Гауссово размытие цветного изображения"""
        return cv2.GaussianBlur(self._image, (kernel_size, kernel_size), sigma).astype(np.uint8)

    def sobel_edge_detection(self) -> NDArray[np.uint8]:
        """Выделение границ оператором Собеля"""
        gray = self.rgb_to_grayscale().astype(np.float32)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        if magnitude.max() > 0:
            magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

        return magnitude

    def gamma_correction(self, gamma: float = 0.5) -> NDArray[np.uint8]:
        """Гамма-коррекция цветного изображения"""
        normalized = self._image.astype(np.float32) / 255.0
        corrected = normalized ** gamma
        return (corrected * 255).astype(np.uint8)

    def histogram_equalization(self) -> NDArray[np.uint8]:
        """Выравнивание гистограммы в LAB пространстве"""
        lab = cv2.cvtColor(self._image, cv2.COLOR_RGB2LAB)

        lightness, channel_a, channel_b = cv2.split(lab)

        lightness_eq = cv2.equalizeHist(lightness)

        lab_eq = cv2.merge([lightness_eq.astype(np.uint8), channel_a, channel_b])

        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB).astype(np.uint8)