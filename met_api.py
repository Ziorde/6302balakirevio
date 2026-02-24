"""
Файл для работы с API
"""

import csv
import json
import os
import random
from typing import Any, Dict, List, Optional, Union

import config

import requests


def setup_directories() -> str:
    """
    Создает необходимую директорию для загрузок

    Returns:
        str: Путь к созданной директории
    """
    download_path = os.path.join(config.PAINTINGS_DIR)
    os.makedirs(download_path, exist_ok=True)
    print(f"Создана директория для загрузок: {download_path}")
    return download_path


def read_paintings_from_csv(csv_filename: str) -> Optional[List[Dict[str, str]]]:
    """
    Читает CSV файл и возвращает список картин

    Args:
        csv_filename: Путь к CSV файлу

    Returns:
        Optional[List[Dict[str, str]]]: Список словарей с данными о картинах
                                       или None в случае ошибки
    """
    if not os.path.exists(csv_filename):
        print(f"Ошибка: Файл {csv_filename} не найден!")
        return None

    paintings = []
    try:
        with open(csv_filename, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('Classification') == 'Paintings':
                    paintings.append({
                        'object_number': row.get('Object Number'),
                        'object_id': row.get('Object ID'),
                        'title': row.get('Title'),
                    })
    except Exception as e:
        print(f"Ошибка при чтении CSV файла: {e}")
        return None

    print(f"Найдено {len(paintings)} картин в коллекции")
    return paintings


def select_random_painting(paintings: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    Выбирает случайную картину из списка

    Args:
        paintings: Список словарей с данными о картинах

    Returns:
        Optional[Dict[str, str]]: Словарь с данными случайной картины
                                 или None в случае ошибки
    """
    if not paintings:
        print("Ошибка: Не найдено картин в коллекции!")
        return None

    random_painting = random.choice(paintings)
    print("\nВыбрана случайная картина:")
    print(f"  Object ID: {random_painting['object_id']}")
    print(f"  Название: {random_painting['title']}")

    return random_painting


def fetch_object_data(object_id: str) -> Optional[Dict[str, Any]]:
    """
    Получает данные о картине из API музея

    Args:
        object_id: Идентификатор объекта в музее

    Returns:
        Optional[Dict[str, Any]]: Данные объекта из API или None в случае ошибки
    """
    api_url = (
        f"https://collectionapi.metmuseum.org/public/collection/v1/objects/"
        f"{object_id}"
    )

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Ошибка при запросе к API: {e}")
        return None


def get_image_url(object_data: Dict[str, Any]) -> Optional[str]:
    """
    Извлекает URL изображения из данных объекта

    Args:
        object_data: Данные объекта из API

    Returns:
        Optional[str]: URL изображения или None, если изображение отсутствует
    """
    image_url = object_data.get('primaryImage')
    if not image_url:
        print("Ошибка: У выбранного объекта нет изображения!")

    return image_url


def download_image(
    image_url: str,
    download_path: str,
    object_id: str,
) -> Optional[str]:
    """
    Скачивает изображение и возвращает путь к сохраненному файлу

    Args:
        image_url: URL изображения
        download_path: Путь для сохранения
        object_id: Идентификатор объекта

    Returns:
        Optional[str]: Путь к сохраненному файлу или None в случае ошибки
    """
    try:
        img_response = requests.get(image_url)
        img_filename = os.path.join(download_path, f"painting_{object_id}.jpg")

        with open(img_filename, 'wb') as f:
            f.write(img_response.content)

        print(f"Изображение сохранено: {img_filename}")
        return img_filename
    except Exception as e:
        print(f"Ошибка при скачивании изображения: {e}")
        return None


def save_metadata(
    object_data: Dict[str, Any],
    download_path: str,
    object_id: str,
) -> Optional[str]:
    """
    Сохраняет метаданные в JSON файл

    Args:
        object_data: Данные объекта из API
        download_path: Путь для сохранения
        object_id: Идентификатор объекта

    Returns:
        Optional[str]: Путь к сохраненному JSON файлу или None в случае ошибки
    """
    try:
        json_filename = os.path.join(download_path, f"painting_{object_id}.json")

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(object_data, f, indent=2, ensure_ascii=False)

        print(f"Метаданные сохранены: {json_filename}")
        return json_filename
    except Exception as e:
        print(f"Ошибка при сохранении метаданных: {e}")
        return None


def download_painting(csv_filename: str) -> Optional[Dict[str, Union[str, int]]]:
    """
    Основная функция для загрузки данных о картине

    Args:
        csv_filename: Путь к CSV файлу с данными о картинах

    Returns:
        Optional[Dict[str, Union[str, int]]]: Словарь с информацией о загруженной картине
                                            или None в случае ошибки
    """
    # Создаем директории
    download_path = setup_directories()

    # Читаем данные из CSV
    paintings = read_paintings_from_csv(csv_filename)
    if paintings is None or not paintings:
        return None

    # Выбираем случайную картину
    random_painting = select_random_painting(paintings)
    if random_painting is None:
        return None

    # Получаем данные из API
    object_data = fetch_object_data(random_painting['object_id'])
    if object_data is None:
        return None

    # Получаем URL изображения
    image_url = get_image_url(object_data)
    if not image_url:
        return None

    # Скачиваем изображение
    img_path = download_image(image_url, download_path, random_painting['object_id'])
    if img_path is None:
        return None

    # Сохраняем метаданные
    json_path = save_metadata(object_data, download_path, random_painting['object_id'])
    if json_path is None:
        return None

    return {
        'img_path': img_path,
        'json_path': json_path,
        'object_id': random_painting['object_id'],
        'title': random_painting['title'],
    }
