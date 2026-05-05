"""
Модуль класса ImageProcessor для управления обработкой изображений
"""
import os
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from models.artwork import Artwork
from models.metadata import Metadata
from models.color_artwork import ColorArtwork
from models.grayscale_artwork import GrayscaleArtwork
from parallel.utils import timer_decorator


class ImageProcessor:
    """
    Класс для управления процессом обработки изображений
    """
    __slots__ = ('_artwork', '_save_dir')

    def __init__(self, save_dir: str = 'data\processed'):
        """
        Инициализация процессора изображений

        Args:
            save_dir: Директория для сохранения результатов
        """
        self._artwork: Optional[Artwork] = None
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._log(f"ImageProcessor инициализирован, директория сохранения: {save_dir}")

    def _log(self, message: str) -> None:
        """Логирование сообщений в консоль"""
        print(f"[ImageProcessor] {message}")

    @property
    def artwork(self) -> Optional[Artwork]:
        """Геттер для текущего произведения искусства"""
        return self._artwork

    @timer_decorator
    def load_from_met(self, csv_filename: str, color: bool = True) -> bool:
        """
        Загрузка случайного изображения из Met Museum API

        Args:
            csv_filename: Путь к CSV файлу с данными
            color: True если хотим изображение в цвете, иначе False

        Returns:
            bool: True если загрузка успешна, иначе False
        """
        from api.met_api import download_painting

        self._log(f"Загрузка изображения из Met Museum (CSV: {csv_filename})")

        result = download_painting(csv_filename)
        if result is None:
            self._log("Ошибка загрузки изображения")
            return False

        image_path = result['img_path']
        image = cv2.imread(image_path)
        if image is None:
            self._log(f"Не удалось загрузить изображение: {image_path}")
            return False

        metadata = Metadata(
            object_id=str(result['object_id']),
            title=result['title']
        )

        self._artwork = ColorArtwork(image, metadata)

        self._log(f"Изображение загружено: {metadata}")
        return True

    @timer_decorator
    def process_rgb_to_grayscale(self, save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: преобразование в полутоновое"""
        if self._artwork is None:
            self._log("Ошибка: изображение не загружено")
            return None

        self._log("Применение преобразования в полутоновое")
        result = self._artwork.rgb_to_grayscale()

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_grayscale.jpg")
            self._artwork.save_processed("grayscale", result)

        return result

    @timer_decorator
    def process_convolution(self, kernel: NDArray[np.float32],
                            save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: свёртка"""
        if self._artwork is None:
            self._log("Ошибка: изображение не загружено")
            return None

        self._log("Применение свёртки")
        result = self._artwork.convolution(kernel)

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_convolution.jpg")
            self._artwork.save_processed("convolution", result)

        return result

    @timer_decorator
    def process_gaussian_blur(self, kernel_size: int = 5, sigma: float = 1.0,
                              save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: гауссово размытие"""
        if self._artwork is None:
            self._log("Ошибка: изображение не загружено")
            return None

        self._log("Применение гауссова размытия")
        result = self._artwork.gaussian_blur(kernel_size, sigma)

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_gaussian.jpg")
            self._artwork.save_processed("gaussian", result)

        return result

    @timer_decorator
    def process_sobel_edge(self, save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: выделение границ оператором Собеля"""
        if self._artwork is None:
            self._log("Ошибка: изображение не загружено")
            return None

        self._log("Применение выделения границ оператором Собеля")
        result = self._artwork.sobel_edge_detection()

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_sobel.jpg")
            self._artwork.save_processed("sobel", result)

        return result

    @timer_decorator
    def process_gamma_correction(self, gamma: float = 0.5,
                                 save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: гамма-коррекция"""
        if self._artwork is None:
            self._log("Ошибка: изображение не загружено")
            return None

        self._log(f"Применение гамма-коррекции (gamma={gamma})")
        result = self._artwork.gamma_correction(gamma)

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_gamma.jpg")
            self._artwork.save_processed("gamma", result)

        return result

    @timer_decorator
    def process_histogram_equalization(self, save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: выравнивание гистограммы"""
        if self._artwork is None:
            self._log("Ошибка: изображение не загружено")
            return None

        self._log("Применение выравнивания гистограммы")
        result = self._artwork.histogram_equalization()

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_histog.jpg")
            self._artwork.save_processed("histog", result)

        return result

    def _save_result(self, image: NDArray[np.uint8], filename: str) -> None:
        """
        Сохранение результата обработки

        Args:
            image: Изображение для сохранения
            filename: Имя файла
        """
        try:
            os.makedirs(self._save_dir, exist_ok=True)
            output_path = os.path.join(self._save_dir, filename)
            cv2.imwrite(output_path, image)
            self._log(f"Результат сохранён: {output_path}")
        except Exception as e:
            self._log(f"Ошибка при сохранении: {e}")

    def process_all(self) -> None:
        """Применение всех методов обработки к текущему изображению"""
        if self._artwork is None:
            self._log("Ошибка: изображение не загружено")
            return

        self._log("Применение всех методов обработки")

        self.process_rgb_to_grayscale()
        self.process_convolution(np.array([[0, -1, 0],
                                           [-1, 5, -1],
                                           [0, -1, 0]]))
        self.process_gaussian_blur()
        self.process_sobel_edge()
        self.process_gamma_correction()
        self.process_histogram_equalization()

    def combine_with(self, other: 'ImageProcessor', save: bool = True) -> Optional[NDArray[np.uint8]]:
        """
        Объединение текущего изображения с другим

        Args:
            other: Другой процессор изображений
            save: Сохранять ли результат

        Returns:
            Optional[NDArray[np.uint8]]: Объединённое изображение
        """
        if self._artwork is None or other.artwork is None:
            self._log("Ошибка: одно из изображений не загружено")
            return None

        self._log("Объединение изображений")
        combined = self._artwork + other.artwork

        if save:
            self._save_result(combined, f"combined_{self._artwork.metadata.object_id}_"
                                        f"{other.artwork.metadata.object_id}.jpg")

        return combined

    def show_info(self) -> None:
        """Вывод информации о текущем изображении"""
        if self._artwork is None:
            self._log("Изображение не загружено")
        else:
            self._log(self._artwork)

    def convert_to_grayscale_artwork(self) -> Optional[GrayscaleArtwork]:
        """
        Конвертация текущего цветного изображения в чёрно-белое

        Returns:
            Optional[GrayscaleArtwork]: Чёрно-белое изображение или None
        """
        if self._artwork is None:
            self._log("Ошибка: изображение не загружено")
            return None

        if isinstance(self._artwork, GrayscaleArtwork):
            self._log("Изображение уже чёрно-белое")
            return self._artwork

        self._log("Конвертация в чёрно-белое изображение")
        gray_image = self._artwork.rgb_to_grayscale()
        return GrayscaleArtwork(gray_image, self._artwork.metadata)