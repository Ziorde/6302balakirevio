"""Асинхронная загрузка изображений"""

import asyncio
import csv
import logging
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import aiohttp
import aiofiles
import cv2
import numpy as np

from .config import (
    REQUEST_TIMEOUT,
    MAX_CONCURRENT_DOWNLOADS,
    MAX_RETRIES
)
from .models.metadata import Metadata

logger = logging.getLogger("metetl.images.async_downloader")


class AsyncImageDownloader:
    """Асинхронный загрузчик изображений из Met Museum API"""

    def __init__(self, csv_path: str, output_dir: Path, max_attempts: int = 10):
        """
        Инициализация асинхронного загрузчика.

        Args:
            csv_path: Путь к CSV файлу
            output_dir: Директория для сохранения
            max_attempts: Максимальное количество попыток загрузки
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.max_attempts = max_attempts
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info(f"AsyncImageDownloader инициализирован. CSV: {csv_path}, "
                    f"output: {output_dir}, max_attempts: {max_attempts}")

    def get_paintings_list(self) -> List[Dict[str, str]]:
        """
        Получение списка картин из CSV.

        Returns:
            List[Dict]: Список словарей с данными о картинах
        """
        logger.debug(f"Чтение списка картин из CSV: {self.csv_path}")

        paintings = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Classification') == 'Paintings':
                    paintings.append({
                        'object_id': row.get('Object ID'),
                        'object_number': row.get('Object Number'),
                        'title': row.get('Title'),
                    })

        logger.info(f"Найдено {len(paintings)} картин в CSV")
        return paintings

    async def fetch_object_data(self, object_id: str) -> Optional[Dict]:
        """
        Асинхронное получение данных объекта из API.

        Args:
            object_id: Идентификатор объекта

        Returns:
            Optional[Dict]: Данные объекта или None
        """
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"

        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(f"Запрос к API для {object_id}, попытка {attempt + 1}/{MAX_RETRIES}")

                async with self.session.get(url, timeout=REQUEST_TIMEOUT) as response:
                    if response.status == 200:
                        logger.debug(f"Успешный ответ API для {object_id}")
                        return await response.json()
                    else:
                        logger.warning(
                            f"Ошибка API для {object_id}: HTTP {response.status}, "
                            f"попытка {attempt + 1}"
                        )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Таймаут при запросе для {object_id}, попытка {attempt + 1}"
                )
            except aiohttp.ClientError as e:
                logger.error(
                    f"Ошибка клиента при запросе для {object_id}: {e}, "
                    f"попытка {attempt + 1}"
                )
            except Exception as e:
                logger.error(
                    f"Неожиданная ошибка при запросе для {object_id}: {e}, "
                    f"попытка {attempt + 1}"
                )

            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1 * (attempt + 1))

        logger.error(f"Не удалось получить данные для {object_id} после {MAX_RETRIES} попыток")
        return None

    async def download_image(
            self,
            index: int,
            paintings_pool: List[Dict],
            used_ids: set
    ) -> Optional[Tuple[int, str, np.ndarray, Metadata]]:
        """
        Асинхронная загрузка одного изображения с повторными попытками.

        Args:
            index: Порядковый номер изображения
            paintings_pool: Полный список доступных картин
            used_ids: Множество уже использованных ID

        Returns:
            Optional[Tuple]: (индекс, object_id, изображение, метаданные)
        """
        async with self.semaphore:
            for attempt in range(1, self.max_attempts + 1):
                available = [p for p in paintings_pool if p['object_id'] not in used_ids]

                if not available:
                    logger.warning(
                        f"Загрузка изображения {index} провалена: "
                        f"не осталось больше картин (попытка {attempt})"
                    )
                    return None

                painting = random.choice(available)
                object_id = painting['object_id']
                used_ids.add(object_id)

                logger.info(
                    f"Загрузка изображения {index} началась "
                    f"(ID: {object_id}), попытка {attempt}/{self.max_attempts}"
                )

                object_data = await self.fetch_object_data(object_id)
                if not object_data:
                    logger.warning(
                        f"Попытка {attempt} для изображения {index} провалена: "
                        f"нет API данных по ID {object_id}"
                    )
                    continue

                image_url = object_data.get('primaryImageSmall')
                if not image_url:
                    logger.warning(
                        f"Попытка {attempt} для изображения {index} провалена: "
                        f"нет URL изображения для ID {object_id}"
                    )
                    continue

                try:
                    async with self.session.get(image_url, timeout=REQUEST_TIMEOUT) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            logger.debug(
                                f"Изображение {index} скачано: {len(image_data)} байт"
                            )
                        else:
                            logger.warning(
                                f"Попытка {attempt} для изображения {index} провалена: "
                                f"HTTP {response.status} при скачивании"
                            )
                            continue
                except Exception as e:
                    logger.warning(
                        f"Попытка {attempt} для изображения {index} провалена: {e}"
                    )
                    continue

                original_filename = f"{index}_{object_id}_original.png"
                original_path = self.output_dir / original_filename

                async with aiofiles.open(original_path, 'wb') as f:
                    await f.write(image_data)

                logger.debug(f"Оригинал сохранен: {original_path}")

                np_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                if image is None:
                    logger.warning(
                        f"Попытка {attempt} для изображения {index} провалена: "
                        f"не удалось декодировать изображение"
                    )
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                metadata = Metadata(
                    object_id=str(object_id),
                    title=painting.get('title', 'Unknown'),
                    artist_name=object_data.get('artistDisplayName'),
                    date=object_data.get('objectDate'),
                    medium=object_data.get('medium'),
                    dimensions=object_data.get('dimensions')
                )

                logger.info(
                    f"Загрузка изображения {index} завершена "
                    f"(ID: {object_id}) после {attempt} попыток"
                )
                return index, object_id, image_rgb, metadata

            logger.error(
                f"Загрузка изображения {index} провалена "
                f"после {self.max_attempts} попыток"
            )
            return None

    async def download_multiple(
            self,
            count: int
    ) -> List[Tuple[int, str, np.ndarray, Metadata]]:
        """
        Асинхронная загрузка нескольких изображений.

        Args:
            count: Количество изображений для загрузки

        Returns:
            List[Tuple]: Список загруженных изображений с сохранением порядка
        """
        logger.info(f"Начинается загрузка {count} изображений")

        paintings = self.get_paintings_list()
        if not paintings:
            logger.error("Нет доступных картин")
            return []

        if count > len(paintings):
            logger.warning(f"Запрошено {count} картин, но доступно только {len(paintings)}")
            count = len(paintings)

        logger.info(
            f"Загрузка {count} изображений из {len(paintings)} доступных "
            f"(максимум попыток на каждое: {self.max_attempts})"
        )

        used_ids = set()

        async with aiohttp.ClientSession() as self.session:
            tasks = [
                self.download_image(idx, paintings, used_ids)
                for idx in range(1, count + 1)
            ]
            results = await asyncio.gather(*tasks)

        valid_results = [r for r in results if r is not None]
        valid_results.sort(key=lambda x: x[0])

        logger.info(f"Успешно загружено {len(valid_results)} из {count} изображений")

        if len(valid_results) < count:
            logger.warning(f"Не удалось загрузить {count - len(valid_results)} изображений")

        return valid_results