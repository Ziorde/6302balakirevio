import os
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CSV_FILE = BASE_DIR.parent / "MetObjects.csv"
OUTPUT_DIR = DATA_DIR / "output"

# Параметры загрузки
REQUEST_TIMEOUT = 30
MAX_CONCURRENT_DOWNLOADS = 5
MAX_RETRIES = 3

# Параметры обработки
MAX_WORKERS = 4  # Количество процессов для параллельной обработки
CONVOLUTION_KERNEL_SIZE = 3

# Доступные методы обработки
PROCESSING_METHODS = {
    '1': {'name': 'convolution', 'description': 'Свёртка (повышение резкости)'},
    '2': {'name': 'gaussian_blur', 'description': 'Гауссово размытие'},
    '3': {'name': 'sobel_edge_detection', 'description': 'Выделение границ (Собель)'},
    '4': {'name': 'gamma_correction', 'description': 'Гамма-коррекция'},
    '5': {'name': 'histogram_equalization', 'description': 'Выравнивание гистограммы'},
    '6': {'name': 'rgb_to_grayscale', 'description': 'Преобразование в полутоновое'},
}

# Создание директорий
os.makedirs(OUTPUT_DIR, exist_ok=True)