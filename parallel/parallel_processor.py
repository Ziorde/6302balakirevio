"""Параллельная обработка изображений с использованием ProcessPoolExecutor"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import OUTPUT_DIR, MAX_WORKERS
from .utils import get_logger
from models.color_artwork import ColorArtwork
from models.grayscale_artwork import GrayscaleArtwork
from models.metadata import Metadata

logger = get_logger(__name__)

class ParallelImageProcessor:
    """Класс для параллельной обработки изображений"""

    def __init__(self, output_dir: str = str(OUTPUT_DIR), max_workers: int = MAX_WORKERS):
        self.output_dir = output_dir
        self.max_workers = max_workers
        os.makedirs(output_dir, exist_ok=True)

    def process_single_image(
            self,
            data: Tuple[int, str, NDArray[np.uint8], dict, str, str]
    ) -> Tuple[int, str, str]:
        """
        Функция для обработки одного изображения в отдельном процессе.

        Args:
            data: Кортеж (индекс, object_id, изображение, метаданные_как_словарь, output_dir, method_name)

        Returns:
            Tuple: (индекс, object_id, путь_к_обработанному)
        """
        index, object_id, image_array, metadata_dict, output_dir, method_name = data

        pid = os.getpid()
        logger.info(f"Обработка '{method_name}' для изображения {index} началась (PID {pid})")

        metadata = Metadata(**metadata_dict)

        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            artwork = ColorArtwork(image_array, metadata)
        else:
            artwork = GrayscaleArtwork(image_array, metadata)

        if method_name == 'convolution':
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)
            processed = artwork.convolution(kernel)
        elif method_name == 'gaussian_blur':
            processed = artwork.gaussian_blur(kernel_size=5, sigma=1.0)
        elif method_name == 'sobel_edge_detection':
            processed = artwork.sobel_edge_detection()
        elif method_name == 'gamma_correction':
            processed = artwork.gamma_correction(gamma=0.5)
        elif method_name == 'histogram_equalization':
            processed = artwork.histogram_equalization()
        elif method_name == 'rgb_to_grayscale':
            processed = artwork.rgb_to_grayscale()
        else:
            logger.warning(f"Неизвестный метод '{method_name}', используется гамма коррекция")
            processed = artwork.gamma_correction()

        processed_filename = f"{index}_{object_id}_{method_name}.png"
        processed_path = os.path.join(output_dir, processed_filename)

        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        cv2.imwrite(processed_path, processed)

        logger.info(f"Обработка '{method_name}' для изображения {index} завершена (PID {pid})")
        return index, object_id, processed_path

    def process_images(
            self,
            images_data: List[Tuple[int, str, np.ndarray, Metadata]],
            method_name: str = 'gamma_correction'
    ) -> List[Tuple[int, str, str]]:
        """
        Параллельная обработка нескольких изображений.

        Args:
            images_data: Список кортежей (индекс, object_id, изображение, метаданные)
            method_name: Название метода обработки

        Returns:
            List[Tuple]: Список результатов (индекс, object_id, путь_к_обработанному)
        """
        if not images_data:
            logger.warning("Нет изображений для обработки")
            return []

        logger.info(f"Начинается параллельная обработка {len(images_data)} изображений")
        logger.info(f"Используемый метод обработки: {method_name}")

        process_data = []
        for index, object_id, image, metadata in images_data:
            metadata_dict = {
                'object_id': metadata.object_id,
                'title': metadata.title,
                'artist_name': metadata.artist_name,
                'date': metadata.date,
                'medium': metadata.medium,
                'dimensions': metadata.dimensions
            }
            process_data.append((
                index, object_id, image, metadata_dict, self.output_dir, method_name
            ))

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.process_single_image, process_data))

        results.sort(key=lambda x: x[0])

        logger.info(f"Обработка завершена, сохранено {len(results)} изображений")
        return results