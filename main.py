import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from met_analysis.generators import (
    chunk_reader,
    culture_filter,
    clean_and_aggregate,
    run_accumulator,
    collect_time_series
)
from met_analysis.analysis import (
    get_top_cultures_from_stats,
    prepare_bar_chart_data,
    get_longest_history_culture,
    prepare_time_series_data
)
from met_analysis.plotting import plot_bar_chart, plot_time_series

import pandas as pd

def main():
    """Основная функция: запуск пайплайна и построение графиков."""

    stats = run_accumulator(
        clean_and_aggregate(
            culture_filter(
                chunk_reader()
            )
        )
    )

    if stats.empty:
        print("Ошибка: нет данных для анализа.")
        return

    top_cultures = get_top_cultures_from_stats(stats)
    print("\nТОП-10 САМЫХ ЧАСТО ВСТРЕЧАЮЩИХСЯ КУЛЬТУР:")
    for i, culture in enumerate(top_cultures.values, 1):
        try:
            count = int(stats.loc[(culture, 'count')])
            print(f"  {i:2d}. {culture:<30} (n={count:,})")
        except KeyError:
            print(f"  {i:2d}. {culture:<30} (n=0)")

    print("\nПодготовка данных для столбцовой диаграммы...")
    bar_df = prepare_bar_chart_data(stats, top_cultures)

    print("\nПостроение столбцовой диаграммы...")
    plot_bar_chart(bar_df, save_path="bar_chart.png")

    longest_history_culture = get_longest_history_culture(stats, top_cultures)
    print(f"\nКультура с самой длительной историей: {longest_history_culture}")

    print("\nПодготовка данных для временного графика...")
    time_data_raw = collect_time_series(longest_history_culture)

    if not time_data_raw.empty:
        time_data_prepared = prepare_time_series_data(
            pd.Series(list(zip(time_data_raw["year"], time_data_raw["age"])))
        )

        print(f"Построение временного графика для культуры: {longest_history_culture}")
        plot_time_series(time_data_prepared, longest_history_culture, save_path="time_series.png")
    else:
        print(f"Недостаточно данных для построения временного графика")

    print("АНАЛИЗ ЗАВЕРШЕН")


if __name__ == "__main__":
    main()