"""
Конфигурация для модуля загрузки и обработки изображений.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_FILE = BASE_DIR / "data" / "MetObjects.csv"
OUTPUT_DIR = DATA_DIR / "output"

REQUEST_TIMEOUT = 30
MAX_CONCURRENT_DOWNLOADS = 5
MAX_RETRIES = 3

MAX_WORKERS = 4
CONVOLUTION_KERNEL_SIZE = 3

PROCESSING_METHODS = {
    '1': {'name': 'convolution', 'description': 'Свёртка (повышение резкости)'},
    '2': {'name': 'gaussian_blur', 'description': 'Гауссово размытие'},
    '3': {'name': 'sobel_edge_detection', 'description': 'Выделение границ (Собель)'},
    '4': {'name': 'gamma_correction', 'description': 'Гамма-коррекция'},
    '5': {'name': 'histogram_equalization', 'description': 'Выравнивание гистограммы'},
    '6': {'name': 'rgb_to_grayscale', 'description': 'Преобразование в полутоновое'},
}

os.makedirs(OUTPUT_DIR, exist_ok=True)