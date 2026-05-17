"""
Модуль статистического анализа данных MetObjects.csv.
"""

import logging
import pandas as pd
from scipy import stats
from typing import Tuple, Optional

from .config import CONFIDENCE_LEVEL, ROLLING_WINDOW
from .generators import (
    chunk_reader,
    culture_filter,
    extract_time_data,
)

logger = logging.getLogger("metetl.analysis.analysis")


def compute_confidence_interval(
    mean: float,
    std: float,
    n: int,
    confidence: float = CONFIDENCE_LEVEL,
) -> Tuple[float, float]:
    """
    Вычисляет доверительный интервал для среднего.

    Args:
        mean: Среднее значение
        std: Стандартное отклонение
        n: Размер выборки
        confidence: Уровень доверия (по умолчанию 0.95)

    Returns:
        Tuple[float, float]: (нижняя граница, верхняя граница)
    """
    if n < 2:
        return (mean, mean)

    sem = std / (n ** 0.5)  # стандартная ошибка среднего
    degrees_freedom = n - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
    margin = t_critical * sem

    return (mean - margin, mean + margin)


def compute_scatter_interval(
    mean: float,
    std: float,
    confidence: float = CONFIDENCE_LEVEL,
) -> Tuple[float, float]:
    """
    Вычисляет интервал рассеяния (prediction interval) для отдельных наблюдений.

    Args:
        mean: Среднее значение
        std: Стандартное отклонение
        confidence: Уровень доверия

    Returns:
        Tuple[float, float]: (нижняя граница, верхняя граница)
    """
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * std

    return (mean - margin, mean + margin)


def get_top_cultures_from_stats(stats: pd.Series, top_n: int = 10) -> pd.Series:
    """
    Возвращает топ-N культур по частоте встречаемости.

    Args:
        stats: Series с мультииндексом (culture, metric)
        top_n: Количество культур для возврата

    Returns:
        pd.Series: Series с индексом (ранг) и значениями (названия культур)
    """
    counts = stats.xs('count', level='metric')
    sorted_cultures = counts.sort_values(ascending=False)
    # Преобразуем индексы в строки, чтобы избежать проблем с float
    top = [str(culture) for culture in sorted_cultures.head(top_n).index.tolist()]

    logger.debug(f"Топ-{top_n} культур по частоте: {top}")

    # Создаем Series с правильным индексом (ранг от 1 до top_n)
    return pd.Series(top, index=range(1, len(top) + 1), name="top_cultures")


def prepare_bar_chart_data(
        stats: pd.Series,
        top_cultures: pd.Series,
) -> pd.DataFrame:
    """
    Подготавливает данные для столбцовой диаграммы.

    Args:
        stats: Series с мультииндексом (culture, metric)
        top_cultures: Series с названиями топ-N культур

    Returns:
        pd.DataFrame со столбцами: culture, mean_age, ci_lower, ci_upper,
                                   scatter_lower, scatter_upper, count
    """
    rows = []

    for culture in top_cultures.values:

        culture_str = str(culture)
        try:
            mean = stats.loc[(culture_str, 'mean')]
            std = stats.loc[(culture_str, 'std')]
            n = int(stats.loc[(culture_str, 'count')])
        except KeyError:
            try:
                mean = stats.loc[(culture, 'mean')]
                std = stats.loc[(culture, 'std')]
                n = int(stats.loc[(culture, 'count')])
            except KeyError:
                logger.warning(f"Пропущена культура: {culture} (нет в статистиках)")
                continue

        if n > 0:
            ci_lower, ci_upper = compute_confidence_interval(mean, std, n)
            sc_lower, sc_upper = compute_scatter_interval(mean, std)

            rows.append({
                "culture": culture_str,
                "mean_age": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "scatter_lower": sc_lower,
                "scatter_upper": sc_upper,
                "count": n
            })

    df = pd.DataFrame(rows)
    logger.debug(f"Подготовлены данные для {len(df)} культур")

    return df


def get_longest_history_culture(
        stats: pd.Series,
        top_cultures: pd.Series,
) -> Optional[str]:
    """
    Возвращает культуру с самой длительной историей (минимальный средний возраст).

    Args:
        stats: Series с мультииндексом (culture, metric)
        top_cultures: Series с названиями топ-N культур

    Returns:
        Optional[str]: Название культуры или None
    """
    valid = []

    for culture in top_cultures.values:
        culture_str = str(culture)
        try:
            mean = stats.loc[(culture_str, 'mean')]
            valid.append((culture_str, mean))
        except KeyError:
            try:
                mean = stats.loc[(culture, 'mean')]
                valid.append((str(culture), mean))
            except KeyError:
                continue

    if not valid:
        logger.warning("Нет валидных культур для определения самой длительной истории")
        return None if top_cultures.empty else str(top_cultures.iloc[0])

    result = min(valid, key=lambda x: x[1])[0]
    logger.info(f"Культура с самой длительной историей: {result}")

    return result


def collect_time_series_for_culture(
    culture: str,
    csv_path: str = None,
) -> pd.DataFrame:
    """
    Собирает временные данные для конкретной культуры.

    Args:
        culture: Название культуры
        csv_path: Путь к CSV файлу (если None, используется из конфига)

    Returns:
        pd.DataFrame с колонками: year, age, rolling_mean
    """
    from .config import CSV_FILE

    if csv_path is None:
        csv_path = CSV_FILE

    logger.info(f"Сбор временных данных для культуры: {culture}")

    years = []
    ages = []

    for chunk in chunk_reader(csv_path):
        for filtered_chunk in culture_filter(chunk, allowed_cultures={culture}):
            for year, age in extract_time_data(filtered_chunk):
                years.append(year)
                ages.append(age)

    if not years:
        logger.warning(f"Нет данных для культуры: {culture}")
        return pd.DataFrame()

    df = pd.DataFrame({"year": years, "age": ages})
    df = df.sort_values("year")

    df_grouped = df.groupby("year", as_index=False)["age"].mean()
    df_grouped["rolling_mean"] = df_grouped["age"].rolling(
        window=ROLLING_WINDOW, min_periods=1
    ).mean()

    logger.info(f"Собрано {len(df_grouped)} точек данных для культуры {culture}")

    return df_grouped