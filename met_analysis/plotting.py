"""Построение графиков с помощью matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .config import FIGURE_SIZE, PLOT_DPI, ROLLING_WINDOW


def set_plot_style():
    """Настройка стиля графиков."""
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 10


def plot_bar_chart(
    df: pd.DataFrame,
    title: str = "Средний возраст объектов при поступлении по культурам",
    save_path: str = "bar_chart.png"
) -> None:
    """
    Строит столбцовую диаграмму со средним возрастом,
    доверительным интервалом и интервалом рассеяния.
    """
    set_plot_style()

    cultures = df["culture"].values
    means = df["mean_age"].values

    # Доверительные интервалы
    ci_errors = [
        [means[i] - df["ci_lower"].iloc[i], df["ci_upper"].iloc[i] - means[i]]
        for i in range(len(df))
    ]
    ci_errors = np.array(ci_errors).T

    # Интервалы рассеяния
    scatter_errors = [
        [means[i] - df["scatter_lower"].iloc[i], df["scatter_upper"].iloc[i] - means[i]]
        for i in range(len(df))
    ]
    scatter_errors = np.array(scatter_errors).T

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=PLOT_DPI)

    x = np.arange(len(cultures))
    width = 0.6

    # Столбцы
    bars = ax.bar(x, means, width, color="steelblue", edgecolor="black", label="Средний возраст")

    # Доверительные интервалы (95% CI)
    ax.errorbar(
        x, means, yerr=ci_errors,
        fmt="none", ecolor="red", capsize=5, capthick=2,
        label=f"95% доверительный интервал"
    )

    # Интервалы рассеяния
    ax.errorbar(
        x, means, yerr=scatter_errors,
        fmt="none", ecolor="orange", capsize=0, capthick=1, alpha=0.5,
        label=f"95% интервал рассеяния"
    )

    # Подписи
    ax.set_xlabel("Культура")
    ax.set_ylabel("Средний возраст при поступлении (годы)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(cultures, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Добавляем значения над столбцами
    for i, (bar, count) in enumerate(zip(bars, df["count"].values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"n={count}",
            ha="center", va="bottom", fontsize=8, rotation=0
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.show()
    print(f"График сохранён: {save_path}")


def plot_time_series(
    df: pd.DataFrame,
    culture: str,
    title: str = None,
    save_path: str = "time_series.png"
) -> None:
    """
    Строит временной график изменения возраста объекта при поступлении
    со скользящим средним.
    """
    if df.empty:
        print("Нет данных для построения временного графика.")
        return

    set_plot_style()

    if title is None:
        title = f"Изменение возраста объектов при поступлении\nКультура: {culture}"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=PLOT_DPI)

    # Исходные данные (усреднённые по годам)
    ax.scatter(df["year"], df["age"], alpha=0.5, s=20, color="gray", label="Средний возраст по годам")

    # Скользящее среднее
    ax.plot(df["year"], df["rolling_mean"], color="red", linewidth=2, label=f"Скользящее среднее (окно={ROLLING_WINDOW})")

    ax.set_xlabel("Год поступления (AccessionYear)")
    ax.set_ylabel("Возраст объекта (годы)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.show()
    print(f"График сохранён: {save_path}")