import os

# Параметры чтения CSV
CHUNK_SIZE = 10000
CSV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "MetObjects.csv")

# Столбцы, необходимые для анализа
REQUIRED_COLUMNS = [
    "Culture",
    "AccessionYear",
    "Object Begin Date"
]

# Типы данных для оптимизации памяти
DTYPES = {
    "Culture": "category",
    "AccessionYear": "float32",
    "Object Begin Date": "float32"
}

# Параметры скользящего среднего
ROLLING_WINDOW = 5

# Параметры доверительных интервалов
CONFIDENCE_LEVEL = 0.95

# Параметры графиков
FIGURE_SIZE = (14, 8)
PLOT_DPI = 100