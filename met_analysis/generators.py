import pandas as pd
import numpy as np
from typing import Generator, Tuple, Optional, Union
from .config import CHUNK_SIZE, CSV_FILE, REQUIRED_COLUMNS


def chunk_reader(
        file_path: str = CSV_FILE,
        chunksize: int = CHUNK_SIZE,
        usecols: list = REQUIRED_COLUMNS,
) -> Generator[pd.DataFrame, None, None]:
    """Генератор 1: чтение CSV по частям."""
    reader = pd.read_csv(
        file_path,
        chunksize=chunksize,
        usecols=usecols,
        dtype=str,
        low_memory=False
    )
    for chunk in reader:
        for col in ["AccessionYear", "Object Begin Date"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].replace(['', '0', 'None', 'null', 'NULL'], pd.NA)
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

        if not chunk.empty:
            yield chunk


def culture_filter(
        source: Union[Generator[pd.DataFrame, None, None], pd.DataFrame],
        allowed_cultures: Optional[set] = None,
        min_year: float = 0,
        max_year: float = float('inf'),
        min_age: float = 0,
        max_age: float = 5000
) -> Generator[pd.DataFrame, None, None]:
    """
    Генератор 2: ПОЛНАЯ фильтрация данных.
    """

    if isinstance(source, pd.DataFrame):
        source = (chunk for chunk in [source])

    for chunk in source:
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
            yield chunk


def clean_and_aggregate(
        source: Generator[pd.DataFrame, None, None]
) -> Generator[Tuple[str, float, float], None, None]:
    """
    Генератор 3: подготовка к агрегации.
    """
    for chunk in source:
        ages = chunk["AccessionYear"] - chunk["Object Begin Date"]

        for culture, age in zip(chunk["Culture"], ages):
            yield (culture, float(age), 1.0)


def run_accumulator(
        source: Generator[Tuple[str, float, float], None, None]
) -> pd.Series:
    """
    Функция-аккумулятор, обрабатывающая поток данных и возвращающая статистику в виде pd.Series.
    """
    stats_series = pd.Series(dtype=float)
    record_count = 0

    for culture, age, _ in source:
        if culture not in stats_series:
            stats_series[culture] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}

        stats_series[culture]["sum"] += age
        stats_series[culture]["sum_sq"] += age * age
        stats_series[culture]["count"] += 1
        record_count += 1

    print(f"Всего обработано записей: {record_count}")

    df = pd.DataFrame(columns=["culture", "metric", "value"])

    for culture, stats in stats_series.items():
        if stats["count"] > 0:
            mean = stats["sum"] / stats["count"]
            variance = (stats["sum_sq"] / stats["count"]) - (mean * mean)
            std = np.sqrt(max(variance, 0))

            new_rows = pd.DataFrame([
                {"culture": culture, "metric": "mean", "value": mean},
                {"culture": culture, "metric": "std", "value": std},
                {"culture": culture, "metric": "count", "value": float(stats["count"])}
            ])
            df = pd.concat([df, new_rows], ignore_index=True)

    if df.empty:
        return pd.Series(dtype=float)

    series = df.set_index(["culture", "metric"])["value"]
    return series


def collect_time_series(
        culture: str
) -> pd.DataFrame:
    """
    Собирает данные для временного графика конкретной культуры.
    """
    source = chunk_reader()
    source = culture_filter(source, allowed_cultures={culture})

    years = []
    ages = []

    for chunk in source:
        ages_chunk = chunk["AccessionYear"] - chunk["Object Begin Date"]
        years.extend(chunk["AccessionYear"].tolist())
        ages.extend(ages_chunk.tolist())

    if not years:
        return pd.DataFrame()

    df = pd.DataFrame({"year": years, "age": ages})
    return df.sort_values("year")


def extract_time_data(
        chunk: pd.DataFrame
) -> Generator[Tuple[float, float], None, None]:
    """
    Генератор для извлечения (AccessionYear, age) из чанка.
    """
    filtered_chunks = list(culture_filter(
        chunk,
        allowed_cultures=None
    ))

    if not filtered_chunks:
        return

    filtered_chunk = filtered_chunks[0]
    ages = filtered_chunk["AccessionYear"] - filtered_chunk["Object Begin Date"]

    for year, age in zip(filtered_chunk["AccessionYear"], ages):
        yield (float(year), float(age))