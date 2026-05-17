"""
Модуль класса ImageProcessor для управления обработкой изображений.
"""
import os
import logging
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from .models.artwork import Artwork
from .models.metadata import Metadata
from .models.color_artwork import ColorArtwork
from .models.grayscale_artwork import GrayscaleArtwork
from metetl.src.metetl.decorators import timer_decorator

logger = logging.getLogger("metetl.images.image_processor")


class ImageProcessor:
    """
    Класс для управления процессом обработки изображений
    """
    __slots__ = ('_artwork', '_save_dir')

    def __init__(self, save_dir: str = 'images/processed'):
        """
        Инициализация процессора изображений

        Args:
            save_dir: Директория для сохранения результатов
        """
        self._artwork: Optional[Artwork] = None
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"ImageProcessor инициализирован, директория сохранения: {save_dir}")

    @property
    def artwork(self) -> Optional[Artwork]:
        """Геттер для текущего произведения искусства"""
        return self._artwork

    @timer_decorator
    def load_from_met(self, csv_filename: str) -> bool:
        """
        Загрузка случайного изображения из Met Museum API

        Args:
            csv_filename: Путь к CSV файлу с данными

        Returns:
            bool: True если загрузка успешна, иначе False
        """
        import requests
        import csv
        import random

        logger.info(f"Загрузка изображения из Met Museum (CSV: {csv_filename})")

        paintings = []
        with open(csv_filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Classification') == 'Paintings':
                    paintings.append(row)

        if not paintings:
            logger.error("Не найдено картин в CSV")
            return False

        painting = random.choice(paintings)
        object_id = painting.get('Object ID')

        logger.info(f"Выбрана картина: {painting.get('Title')} (ID: {object_id})")

        api_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"

        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            object_data = response.json()
        except Exception as e:
            logger.error(f"Ошибка при запросе к API: {e}")
            return False

        image_url = object_data.get('primaryImageSmall')
        if not image_url:
            logger.error("У объекта нет изображения")
            return False

        try:
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()

            img_path = os.path.join(self._save_dir, f"painting_{object_id}.jpg")
            with open(img_path, 'wb') as f:
                f.write(img_response.content)

            image = cv2.imread(img_path)
            if image is None:
                logger.error(f"Не удалось загрузить изображение: {img_path}")
                return False

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            metadata = Metadata(
                object_id=str(object_id),
                title=painting.get('Title', 'Unknown'),
                artist_name=object_data.get('artistDisplayName'),
                date=object_data.get('objectDate'),
                medium=object_data.get('medium'),
                dimensions=object_data.get('dimensions')
            )

            self._artwork = ColorArtwork(image_rgb, metadata)

            logger.info(f"Изображение загружено: {metadata}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при скачивании изображения: {e}")
            return False

    @timer_decorator
    def process_rgb_to_grayscale(self, save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: преобразование в полутоновое"""
        if self._artwork is None:
            logger.error("Ошибка: изображение не загружено")
            return None

        logger.info("Применение преобразования в полутоновое")
        result = self._artwork.rgb_to_grayscale()

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_grayscale.jpg")
            self._artwork.save_processed("grayscale", result)

        return result

    @timer_decorator
    def process_convolution(self, kernel: NDArray[np.float32], save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: свёртка"""
        if self._artwork is None:
            logger.error("Ошибка: изображение не загружено")
            return None

        logger.info("Применение свёртки")
        result = self._artwork.convolution(kernel)

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_convolution.jpg")
            self._artwork.save_processed("convolution", result)

        return result

    @timer_decorator
    def process_gaussian_blur(self, kernel_size: int = 5, sigma: float = 1.0, save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: гауссово размытие"""
        if self._artwork is None:
            logger.error("Ошибка: изображение не загружено")
            return None

        logger.info("Применение гауссова размытия")
        result = self._artwork.gaussian_blur(kernel_size, sigma)

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_gaussian.jpg")
            self._artwork.save_processed("gaussian", result)

        return result

    @timer_decorator
    def process_sobel_edge(self, save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: выделение границ оператором Собеля"""
        if self._artwork is None:
            logger.error("Ошибка: изображение не загружено")
            return None

        logger.info("Применение выделения границ оператором Собеля")
        result = self._artwork.sobel_edge_detection()

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_sobel.jpg")
            self._artwork.save_processed("sobel", result)

        return result

    @timer_decorator
    def process_gamma_correction(self, gamma: float = 0.5, save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: гамма-коррекция"""
        if self._artwork is None:
            logger.error("Ошибка: изображение не загружено")
            return None

        logger.info(f"Применение гамма-коррекции (gamma={gamma})")
        result = self._artwork.gamma_correction(gamma)

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_gamma.jpg")
            self._artwork.save_processed("gamma", result)

        return result

    @timer_decorator
    def process_histogram_equalization(self, save: bool = True) -> Optional[NDArray[np.uint8]]:
        """Обработка: выравнивание гистограммы"""
        if self._artwork is None:
            logger.error("Ошибка: изображение не загружено")
            return None

        logger.info("Применение выравнивания гистограммы")
        result = self._artwork.histogram_equalization()

        if save:
            self._save_result(result, f"{self._artwork.metadata.object_id}_histogram.jpg")
            self._artwork.save_processed("histogram", result)

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

            # Конвертируем RGB в BGR для сохранения через OpenCV
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image

            cv2.imwrite(output_path, image_bgr)
            logger.info(f"Результат сохранён: {output_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении: {e}")

    def process_all(self) -> None:
        """Применение всех методов обработки к текущему изображению"""
        if self._artwork is None:
            logger.error("Ошибка: изображение не загружено")
            return

        logger.info("Применение всех методов обработки")

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
            logger.error("Ошибка: одно из изображений не загружено")
            return None

        logger.info("Объединение изображений")
        combined = self._artwork + other.artwork

        if save:
            self._save_result(
                combined,
                f"combined_{self._artwork.metadata.object_id}_"
                f"{other.artwork.metadata.object_id}.jpg"
            )

        return combined

    def show_info(self) -> None:
        """Вывод информации о текущем изображении"""
        if self._artwork is None:
            logger.warning("Изображение не загружено")
        else:
            logger.info(str(self._artwork))

    def convert_to_grayscale_artwork(self) -> Optional[GrayscaleArtwork]:
        """
        Конвертация текущего цветного изображения в чёрно-белое

        Returns:
            Optional[GrayscaleArtwork]: Чёрно-белое изображение или None
        """
        if self._artwork is None:
            logger.error("Ошибка: изображение не загружено")
            return None

        if isinstance(self._artwork, GrayscaleArtwork):
            logger.info("Изображение уже чёрно-белое")
            return self._artwork

        logger.info("Конвертация в чёрно-белое изображение")
        gray_image = self._artwork.rgb_to_grayscale()

        return GrayscaleArtwork(gray_image, self._artwork.metadata)