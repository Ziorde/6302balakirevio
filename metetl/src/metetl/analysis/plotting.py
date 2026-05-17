"""
Модуль построения графиков для анализа данных.
"""

import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURE_SIZE, PLOT_DPI, ROLLING_WINDOW

logger = logging.getLogger("metetl.analysis.plotting")


def set_plot_style() -> None:
    """Настройка стиля графиков."""
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.figsize"] = FIGURE_SIZE
    plt.rcParams["figure.dpi"] = PLOT_DPI
    plt.rcParams["savefig.dpi"] = PLOT_DPI
    plt.rcParams["savefig.bbox"] = "tight"


def plot_bar_chart(
        df: pd.DataFrame,
        title: str = "Средний возраст объектов при поступлении по культурам",
        save_path: str = "bar_chart.png",
) -> None:
    """
    Строит столбцовую диаграмму со средним возрастом,
    доверительным интервалом и интервалом рассеяния.

    Args:
        df: DataFrame с данными для диаграммы
        title: Заголовок диаграммы
        save_path: Путь для сохранения
    """
    logger.info(f"Построение столбцовой диаграммы: {save_path}")

    set_plot_style()

    cultures = df["culture"].values
    means = df["mean_age"].values

    ci_lower_errors = means - df["ci_lower"].values
    ci_upper_errors = df["ci_upper"].values - means
    ci_errors = np.array([ci_lower_errors, ci_upper_errors])

    scatter_lower_errors = means - df["scatter_lower"].values
    scatter_upper_errors = df["scatter_upper"].values - means
    scatter_errors = np.array([scatter_lower_errors, scatter_upper_errors])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    x = np.arange(len(cultures))
    width = 0.6

    bars = ax.bar(
        x, means, width,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
        label="Средний возраст"
    )

    ax.errorbar(
        x, means, yerr=ci_errors,
        fmt="none", ecolor="red", capsize=5, capthick=2,
        label="95% доверительный интервал"
    )

    ax.errorbar(
        x, means, yerr=scatter_errors,
        fmt="none", ecolor="orange", capsize=0, capthick=1,
        alpha=0.5, label="95% интервал рассеяния"
    )

    ax.set_xlabel("Культура")
    ax.set_ylabel("Средний возраст при поступлении (годы)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(cultures, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for i, (bar, count) in enumerate(zip(bars, df["count"].values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"n={count}",
            ha="center", va="bottom",
            fontsize=8, rotation=0
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Столбцовая диаграмма сохранена: {save_path}")


def plot_time_series(
        df: pd.DataFrame,
        culture: str,
        title: str = None,
        save_path: str = "time_series.png",
) -> None:
    """
    Строит временной график изменения возраста объекта при поступлении
    со скользящим средним.

    Args:
        df: DataFrame с колонками year, age, rolling_mean
        culture: Название культуры
        title: Заголовок графика
        save_path: Путь для сохранения
    """
    if df.empty:
        logger.warning(f"Нет данных для построения временного графика для культуры: {culture}")
        return

    logger.info(f"Построение временного графика для культуры {culture}: {save_path}")

    set_plot_style()

    if title is None:
        title = f"Изменение возраста объектов при поступлении\nКультура: {culture}"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.scatter(
        df["year"], df["age"],
        alpha=0.5, s=20,
        color="gray",
        label="Средний возраст по годам"
    )

    ax.plot(
        df["year"], df["rolling_mean"],
        color="red", linewidth=2,
        label=f"Скользящее среднее (окно={ROLLING_WINDOW})"
    )

    ax.set_xlabel("Год поступления (AccessionYear)")
    ax.set_ylabel("Возраст объекта (годы)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Временной график сохранен: {save_path}")