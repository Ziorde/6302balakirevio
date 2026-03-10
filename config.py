"""
Файл с настройками проекта
"""

import numpy as np


# Пути к директориям

# Для скаченных картин
PAINTINGS_DIR = 'paintings'
# Для обработанных файлов
SAVE_DIR = 'processed'


# Цветовые константы для приведения цветного изображения к полутоновому
GRAYSCALE_WEIGHTS = {
    'red': 0.2989,
    'green': 0.5870,
    'blue': 0.1140,
}

# Настройки обработки
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 1.0
GAMMA_VALUE = 0.5

# Ядра для свертки
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

SHARPNESS = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
