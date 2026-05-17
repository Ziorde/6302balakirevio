"""
Генераторы для пайплайна обработки данных MetObjects.csv.
"""

import logging
import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional, Union, Set

from .config import CHUNK_SIZE, CSV_FILE, REQUIRED_COLUMNS

logger = logging.getLogger("metetl.analysis.generators")


def chunk_reader(
        file_path: str = CSV_FILE,
        chunksize: int = CHUNK_SIZE,
        usecols: list = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Генератор 1: чтение CSV по частям.

    Args:
        file_path: Путь к CSV файлу
        chunksize: Размер чанка
        usecols: Список колонок для чтения

    Yields:
        pd.DataFrame: Чанк данных
    """
    if usecols is None:
        usecols = REQUIRED_COLUMNS

    logger.debug(f"Начало чтения файла: {file_path}, размер чанка: {chunksize}")

    reader = pd.read_csv(
        file_path,
        chunksize=chunksize,
        usecols=usecols,
        dtype=str,
        low_memory=False
    )

    chunk_count = 0
    for chunk in reader:
        for col in ["AccessionYear", "Object Begin Date"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].replace(['', '0', 'None', 'null', 'NULL'], pd.NA)
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

        if not chunk.empty:
            chunk_count += 1
            logger.debug(f"Прочитан чанк {chunk_count}, строк: {len(chunk)}")
            yield chunk

    logger.debug(f"Всего прочитано чанков: {chunk_count}")


def culture_filter(
        source: Union[Generator[pd.DataFrame, None, None], pd.DataFrame],
        allowed_cultures: Optional[Set[str]] = None,
        min_year: float = 0,
        max_year: float = float('inf'),
        min_age: float = 0,
        max_age: float = 5000,
) -> Generator[pd.DataFrame, None, None]:
    """
    Генератор 2: фильтрация данных.

    Args:
        source: Источник данных (генератор чанков или DataFrame)
        allowed_cultures: Множество разрешенных культур (None - все)
        min_year: Минимальный год поступления
        max_year: Максимальный год поступления
        min_age: Минимальный возраст объекта
        max_age: Максимальный возраст объекта

    Yields:
        pd.DataFrame: Отфильтрованный чанк данных
    """
    logger.debug("Начало фильтрации данных")

    if isinstance(source, pd.DataFrame):
        chunks = [source]
    else:
        chunks = source

    filtered_count = 0

    for chunk in chunks:
        original_size = len(chunk)

        if allowed_cultures is not None:
            chunk = chunk[chunk["Culture"].isin(allowed_cultures)]

        chunk = chunk.dropna(subset=["AccessionYear", "Object Begin Date"])

        chunk = chunk[(chunk["AccessionYear"] > min_year) & (chunk["AccessionYear"] <= max_year)]

        chunk = chunk[chunk["Object Begin Date"] > min_year]

        chunk = chunk[chunk["AccessionYear"] >= chunk["Object Begin Date"]]

        ages = chunk["AccessionYear"] - chunk["Object Begin Date"]
        age_mask = (ages >= min_age) & (ages <= max_age)
        chunk = chunk[age_mask]

        if not chunk.empty:
            filtered_count += len(chunk)
            logger.debug(
                f"Отфильтровано: {original_size} -> {len(chunk)} строк "
                f"(всего отфильтровано: {filtered_count})"
            )
            yield chunk

    logger.debug(f"Фильтрация завершена. Всего отфильтровано строк: {filtered_count}")


def clean_and_aggregate(
        source: Generator[pd.DataFrame, None, None]
) -> Generator[Tuple[str, float, float], None, None]:
    """
    Генератор 3: подготовка данных к агрегации.

    Args:
        source: Отфильтрованные чанки данных

    Yields:
        Tuple[str, float, float]: (культура, возраст, вес)
    """
    logger.debug("Начало подготовки данных к агрегации")

    record_count = 0

    for chunk in source:
        ages = chunk["AccessionYear"] - chunk["Object Begin Date"]

        for culture, age in zip(chunk["Culture"], ages):
            record_count += 1
            yield (culture, float(age), 1.0)

    logger.debug(f"Подготовлено записей для агрегации: {record_count}")


def run_accumulator(
        source: Generator[Tuple[str, float, float], None, None]
) -> pd.Series:
    """
    Функция-аккумулятор, обрабатывающая поток данных и возвращающая статистику.

    Args:
        source: Поток данных (культура, возраст, вес)

    Returns:
        pd.Series: Статистики с мультииндексом (culture, metric)
    """
    logger.info("Начало аккумуляции статистик")

    stats_dict = {}
    record_count = 0

    for culture, age, _ in source:
        if culture not in stats_dict:
            stats_dict[culture] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}

        stats_dict[culture]["sum"] += age
        stats_dict[culture]["sum_sq"] += age * age
        stats_dict[culture]["count"] += 1
        record_count += 1

        if record_count % 100000 == 0:
            logger.debug(f"Обработано записей: {record_count}")

    logger.info(f"Всего обработано записей: {record_count}")
    logger.info(f"Уникальных культур: {len(stats_dict)}")

    rows = []
    for culture, stats in stats_dict.items():
        if stats["count"] > 0:
            mean = stats["sum"] / stats["count"]
            variance = (stats["sum_sq"] / stats["count"]) - (mean * mean)
            std = np.sqrt(max(variance, 0))

            rows.append({
                "culture": culture,
                "metric": "mean",
                "value": mean
            })
            rows.append({
                "culture": culture,
                "metric": "std",
                "value": std
            })
            rows.append({
                "culture": culture,
                "metric": "count",
                "value": float(stats["count"])
            })

    if not rows:
        logger.warning("Нет данных для формирования статистик")
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows)
    series = df.set_index(["culture", "metric"])["value"]

    logger.info(f"Аккумуляция завершена. Сформировано {len(series)} записей статистик")
    return series


def collect_time_series(
        culture: str,
        csv_path: str = CSV_FILE,
) -> pd.DataFrame:
    """
    Сбор временных данных для конкретной культуры.

    Args:
        culture: Название культуры
        csv_path: Путь к CSV файлу

    Returns:
        pd.DataFrame: DataFrame с колонками year и age
    """
    logger.debug(f"Сбор временных данных для культуры: {culture}")

    source = chunk_reader(csv_path)
    filtered = culture_filter(source, allowed_cultures={culture})

    years = []
    ages = []

    for chunk in filtered:
        ages_chunk = chunk["AccessionYear"] - chunk["Object Begin Date"]
        years.extend(chunk["AccessionYear"].tolist())
        ages.extend(ages_chunk.tolist())

    if not years:
        logger.warning(f"Нет данных для культуры: {culture}")
        return pd.DataFrame()

    df = pd.DataFrame({"year": years, "age": ages})
    logger.debug(f"Собрано {len(df)} записей для культуры {culture}")

    return df.sort_values("year")


def extract_time_data(
        chunk: pd.DataFrame,
) -> Generator[Tuple[float, float], None, None]:
    """
    Генератор для извлечения временных данных из чанка.

    Args:
        chunk: DataFrame с данными

    Yields:
        Tuple[float, float]: (год поступления, возраст объекта)
    """
    filtered_chunks = list(culture_filter(chunk, allowed_cultures=None))

    if not filtered_chunks:
        return

    filtered_chunk = filtered_chunks[0]
    ages = filtered_chunk["AccessionYear"] - filtered_chunk["Object Begin Date"]

    for year, age in zip(filtered_chunk["AccessionYear"], ages):
        yield (float(year), float(age))