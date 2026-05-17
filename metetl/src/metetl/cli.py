"""
Модуль интерфейса командной строки для metetl.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .logging_config import setup_logging, get_logger

logger = get_logger("cli")


def prepare_command(args: argparse.Namespace) -> None:
    """
    Подготовка JSON файла с метаданными выбранных для скачивания изображений.

    Args:
        args: Аргументы командной строки
    """
    import csv

    csv_path = Path(args.csv)
    output_path = Path(args.output)

    if not csv_path.exists():
        logger.error(f"CSV файл не найден: {csv_path}")
        sys.exit(1)

    logger.info(f"Чтение CSV файла: {csv_path}")

    paintings = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Classification') == 'Paintings':
                paintings.append({
                    'object_id': row.get('Object ID'),
                    'object_number': row.get('Object Number'),
                    'title': row.get('Title'),
                    'artist': row.get('Artist Display Name'),
                    'date': row.get('Object Date'),
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(paintings, f, indent=2, ensure_ascii=False)

    logger.info(f"Метаданные сохранены в {output_path}. Найдено {len(paintings)} картин.")


async def process_command_async(args: argparse.Namespace) -> None:
    """
    Асинхронная загрузка и обработка изображений.

    Args:
        args: Аргументы командной строки
    """
    from .images.async_downloader import AsyncImageDownloader
    from .images.parallel_processor import ParallelImageProcessor

    input_path = Path(args.input)
    output_dir = Path(args.output)
    num_images = args.num

    if not input_path.exists():
        logger.error(f"JSON файл не найден: {input_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path("data/MetObjects.csv")

    logger.info(f"Запуск загрузки {num_images} изображений")
    logger.info(f"Результаты будут сохранены в: {output_dir}")

    downloader = AsyncImageDownloader(str(csv_path), output_dir)
    images_data = await downloader.download_multiple(num_images)

    if not images_data:
        logger.error("Не удалось загрузить ни одного изображения")
        return

    logger.info(f"Загружено изображений: {len(images_data)}")

    processor = ParallelImageProcessor(output_dir=str(output_dir))
    results = processor.process_images(images_data, method_name='gamma_correction')

    logger.info("Результаты обработки:")
    for index, object_id, path in results:
        logger.info(f"  {index}. ID: {object_id} -> {Path(path).name}")

    logger.info(f"Всего обработано: {len(results)} изображений")


def process_command(args: argparse.Namespace) -> None:
    """
    Запуск процесса загрузки и обработки изображений.

    Args:
        args: Аргументы командной строки
    """
    asyncio.run(process_command_async(args))


def analyze_command(args: argparse.Namespace) -> None:
    """
    Запуск анализа датасета из CSV файла.

    Args:
        args: Аргументы командной строки
    """
    from .analysis.generators import chunk_reader, culture_filter, clean_and_aggregate, run_accumulator
    from .analysis.analysis import (
        get_top_cultures_from_stats,
        prepare_bar_chart_data,
        get_longest_history_culture,
        collect_time_series_for_culture
    )
    from .analysis.plotting import plot_bar_chart, plot_time_series

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        logger.error(f"CSV файл не найден: {csv_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Запуск анализа данных из {csv_path}")
    logger.info(f"Результаты будут сохранены в: {output_dir}")

    logger.info("Чтение данных...")
    chunk_gen = chunk_reader(str(csv_path))
    filtered_gen = culture_filter(chunk_gen)
    aggregated_gen = clean_and_aggregate(filtered_gen)
    stats = run_accumulator(aggregated_gen)

    logger.info("Вычисление статистик...")

    top_cultures = get_top_cultures_from_stats(stats)

    top_cultures_names = [str(name) for name in top_cultures.values]
    logger.info(f"Топ-10 культур: {', '.join(top_cultures_names)}")

    bar_data = prepare_bar_chart_data(stats, top_cultures)

    bar_chart_path = output_dir / "bar_chart.png"
    plot_bar_chart(
        bar_data,
        title="Средний возраст объектов при поступлении по культурам",
        save_path=str(bar_chart_path)
    )

    longest_culture = get_longest_history_culture(stats, top_cultures)
    logger.info(f"Культура с самой длительной историей: {longest_culture}")

    time_series_data = collect_time_series_for_culture(str(longest_culture), str(csv_path))
    if not time_series_data.empty:
        time_series_path = output_dir / f"time_series_{longest_culture}.png"
        safe_filename = str(longest_culture).replace('/', '_').replace('\\', '_').replace(':', '_')
        time_series_path = output_dir / f"time_series_{safe_filename}.png"
        plot_time_series(
            time_series_data,
            str(longest_culture),
            title=f"Изменение возраста объектов при поступлении\nКультура: {longest_culture}",
            save_path=str(time_series_path)
        )
    else:
        logger.warning(f"Нет временных данных для культуры: {longest_culture}")

    logger.info("Анализ завершен")
    logger.info(f"Графики сохранены в: {output_dir}")


def create_parser() -> argparse.ArgumentParser:
    """
    Создание парсера аргументов командной строки.

    Returns:
        argparse.ArgumentParser: Настроенный парсер
    """
    parser = argparse.ArgumentParser(
        prog="metetl",
        description="Metropolitan Museum of Art ETL Pipeline - "
                    "скачивание, обработка изображений и анализ данных",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  metetl prepare --csv data/MetObjects.csv --output data/to_download.json
  metetl process --input data/to_download.json --output images --num 5
  metetl analyze --csv data/MetObjects.csv --output-dir data/plots
  metetl --help
        """
    )

    parser.add_argument(
        "--log-config",
        default=None,
        help="Путь к JSON файлу конфигурации логирования"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Уровень логирования в консоль (по умолчанию: INFO)"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Доступные команды"
    )

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Подготовка JSON файла с метаданными изображений"
    )
    prepare_parser.add_argument(
        "--csv",
        required=True,
        help="Путь к CSV файлу MetObjects.csv"
    )
    prepare_parser.add_argument(
        "--output",
        default="data/to_download.json",
        help="Путь для сохранения JSON файла (по умолчанию: data/to_download.json)"
    )

    process_parser = subparsers.add_parser(
        "process",
        help="Загрузка и обработка изображений"
    )
    process_parser.add_argument(
        "--input",
        required=True,
        help="Путь к JSON файлу с метаданными"
    )
    process_parser.add_argument(
        "--output",
        default="images",
        help="Директория для сохранения изображений (по умолчанию: images)"
    )
    process_parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Количество изображений для загрузки (по умолчанию: 5)"
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Анализ данных из CSV файла"
    )
    analyze_parser.add_argument(
        "--csv",
        required=True,
        help="Путь к CSV файлу MetObjects.csv"
    )
    analyze_parser.add_argument(
        "--output-dir",
        default="data/plots",
        help="Директория для сохранения графиков (по умолчанию: data/plots)"
    )

    return parser


def main() -> None:
    """
    Главная точка входа CLI.
    """
    parser = create_parser()
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)

    if args.log_config:
        setup_logging(config_path=args.log_config, console_level=log_level)
    else:
        setup_logging(console_level=log_level)

    logger.info("=" * 60)
    logger.info("Metropolitan Museum of Art ETL Pipeline")
    logger.info("=" * 60)

    if args.command == "prepare":
        logger.info("Выполнение команды: prepare")
        prepare_command(args)
    elif args.command == "process":
        logger.info("Выполнение команды: process")
        process_command(args)
    elif args.command == "analyze":
        logger.info("Выполнение команды: analyze")
        analyze_command(args)
    else:
        parser.print_help()
        logger.warning("Не указана команда. Используйте --help для справки.")


if __name__ == "__main__":
    main()