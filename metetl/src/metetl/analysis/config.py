"""
Конфигурация для модуля анализа данных.
"""

import os

CHUNK_SIZE = 10000
CSV_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "MetObjects.csv"
)

REQUIRED_COLUMNS = [
    "Culture",
    "AccessionYear",
    "Object Begin Date"
]

DTYPES = {
    "Culture": "category",
    "AccessionYear": "float32",
    "Object Begin Date": "float32"
}

ROLLING_WINDOW = 5
CONFIDENCE_LEVEL = 0.95

FIGURE_SIZE = (14, 8)
PLOT_DPI = 100