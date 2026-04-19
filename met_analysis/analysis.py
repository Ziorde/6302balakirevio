import pandas as pd
from scipy import stats
from typing import Tuple, List
from .config import CONFIDENCE_LEVEL, ROLLING_WINDOW


def compute_confidence_interval(
    mean: float,
    std: float,
    n: int,
    confidence: float = CONFIDENCE_LEVEL
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
    confidence: float = CONFIDENCE_LEVEL
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


def get_top_cultures_from_stats(
    stats: pd.Series
) -> pd.Series:
    """
    Возвращает топ-10 культур по частоте встречаемости как Series.

    Args:
        stats: Series с мультииндексом (culture, metric)

    Returns:
        pd.Series: Series с индексом (ранг) и значениями (названия культур)
    """
    # Извлекаем count для каждой культуры
    counts = stats.xs('count', level='metric')
    sorted_cultures = counts.sort_values(ascending=False)
    top = sorted_cultures.head(10).index.tolist()
    return pd.Series(top, name="top_cultures")


def prepare_bar_chart_data(
    stats: pd.Series,
    top_cultures: pd.Series
) -> pd.DataFrame:
    """
    Подготавливает данные для столбцовой диаграммы.

    Args:
        stats: Series с мультииндексом (culture, metric)
        top_cultures: Series с названиями топ-10 культур

    Returns:
        pd.DataFrame со столбцами: culture, mean_age, ci_lower, ci_upper,
                                   scatter_lower, scatter_upper, count
    """
    rows = []
    for culture in top_cultures.values:
        try:
            mean = stats.loc[(culture, 'mean')]
            std = stats.loc[(culture, 'std')]
            n = int(stats.loc[(culture, 'count')])
        except KeyError:
            continue

        if n > 0:
            ci_lower, ci_upper = compute_confidence_interval(mean, std, n)
            sc_lower, sc_upper = compute_scatter_interval(mean, std)

            rows.append({
                "culture": culture,
                "mean_age": mean,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "scatter_lower": sc_lower,
                "scatter_upper": sc_upper,
                "count": n
            })

    return pd.DataFrame(rows)


def get_longest_history_culture(
    stats: pd.Series,
    top_cultures: pd.Series
) -> str:
    """
    Возвращает культуру с самой длительной историей (минимальный средний возраст).

    Args:
        stats: Series с мультииндексом (culture, metric)
        top_cultures: Series с названиями топ-10 культур

    Returns:
        str: Название культуры
    """
    valid = []
    for culture in top_cultures.values:
        try:
            mean = stats.loc[(culture, 'mean')]
            valid.append((culture, mean))
        except KeyError:
            continue

    if not valid:
        return top_cultures.iloc[0] if not top_cultures.empty else None

    return min(valid, key=lambda x: x[1])[0]


def prepare_time_series_data(
        chunked_data: pd.Series
) -> pd.DataFrame:
    """
    Подготавливает данные для временного графика.

    Args:
        chunked_data: Series с кортежами (AccessionYear, age) или список

    Returns:
        pd.DataFrame с колонками: year, age, rolling_mean
    """
    if chunked_data.empty if hasattr(chunked_data, 'empty') else not chunked_data:
        return pd.DataFrame()

    if isinstance(chunked_data, pd.Series):
        chunked_data = chunked_data.tolist()

    df = pd.DataFrame(chunked_data, columns=["year", "age"])
    df = df.dropna().sort_values("year")

    df_grouped = df.groupby("year", as_index=False)["age"].mean()

    df_grouped["rolling_mean"] = df_grouped["age"].rolling(
        window=ROLLING_WINDOW, min_periods=1
    ).mean()

    return df_grouped


def collect_time_series_for_culture(
        culture: str,
) -> pd.DataFrame:
    """
    Собирает данные для временного графика конкретной культуры.
    Использует пайплайн генераторов.

    Args:
        culture: Название культуры

    Returns:
        pd.DataFrame с колонками: year, age, rolling_mean
    """
    from met_analysis.generators import chunk_reader, culture_filter, extract_time_data

    years = []
    ages = []

    # Пайплайн для сбора временных данных
    for chunk in chunk_reader():
        for filtered_chunk in culture_filter(chunk, allowed_cultures={culture}):
            for year, age in extract_time_data(filtered_chunk):
                years.append(year)
                ages.append(age)

    if not years:
        return pd.DataFrame()

    df = pd.DataFrame({"year": years, "age": ages})
    df = df.sort_values("year")

    # Группировка по годам и скользящее среднее
    df_grouped = df.groupby("year", as_index=False)["age"].mean()
    df_grouped["rolling_mean"] = df_grouped["age"].rolling(
        window=ROLLING_WINDOW, min_periods=1
    ).mean()

    return df_grouped