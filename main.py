"""Точка входа для лабораторной работы №4"""

import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parallel.config import OUTPUT_DIR, CSV_FILE, PROCESSING_METHODS
from parallel.async_downloader import AsyncImageDownloader
from parallel.parallel_processor import ParallelImageProcessor
from parallel.utils import get_logger, timer_decorator

logger = get_logger(__name__)

def print_processing_methods() -> str:
    """
    Выводит список доступных методов обработки и запрашивает выбор пользователя.

    Returns:
        str: Название выбранного метода
    """
    print("Доступные методы обработки изображений:")

    for key, method in PROCESSING_METHODS.items():
        print(f"  {key}. {method['description']} (метод: {method['name']})")

    print("="*60)

    while True:
        choice = input("\nВыберите номер метода обработки (по умолчанию 4 - гамма-коррекция): ").strip()

        if choice == "":
            selected = PROCESSING_METHODS['4']
            print(f"Выбрано: {selected['description']}")
            return selected['name']

        if choice in PROCESSING_METHODS:
            selected = PROCESSING_METHODS[choice]
            print(f"Выбрано: {selected['description']}")
            return selected['name']

        print(f"Ошибка: введите число от 1 до {len(PROCESSING_METHODS)}")


def parse_arguments():
    """
    Парсинг аргументов командной строки.

    Returns:
        tuple: (count, use_pipeline, method_name)
    """
    if len(sys.argv) < 2:
        print("Использование: python main.py <количество_изображений> [--method METHOD_NAME]")
        print("  --method NAME     - указать метод обработки (convolution, gaussian_blur, sobel_edge_detection,")
        print("                      gamma_correction, histogram_equalization, rgb_to_grayscale)")
        print("  --interactive     - интерактивный выбор метода обработки")
        sys.exit(1)

    try:
        count = int(sys.argv[1])
    except ValueError:
        print("Ошибка: количество изображений должно быть целым числом")
        sys.exit(1)

    interactive = "--interactive" in sys.argv

    method_name = 'gamma_correction'
    for i, arg in enumerate(sys.argv):
        if arg == "--method" and i + 1 < len(sys.argv):
            method_name = sys.argv[i + 1]
            valid_methods = [m['name'] for m in PROCESSING_METHODS.values()]
            if method_name not in valid_methods:
                print(f"Ошибка: неизвестный метод '{method_name}'")
                print(f"Доступные методы: {', '.join(valid_methods)}")
                sys.exit(1)
            break

    return count, method_name, interactive


@timer_decorator
async def main():
    count, method_name, interactive = parse_arguments()

    if interactive:
        method_name = print_processing_methods()
    else:
        if not any("--method" in arg for arg in sys.argv):
            print(f"\nИспользуется метод по умолчанию: {method_name}")
            print("Для выбора другого метода используйте --interactive или --method METHOD_NAME\n")
    print(f"Запуск асинхронной загрузки и параллельной обработки")
    print(f"Количество изображений: {count}")
    print(f"Метод обработки: {method_name}")

    run_output_dir = OUTPUT_DIR / f"run_{int(time.time())}_{method_name}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Результаты будут сохранены в: {run_output_dir}")

    downloader = AsyncImageDownloader(str(CSV_FILE), run_output_dir)
    images_data = await downloader.download_multiple(count)

    if not images_data:
        logger.error("Не удалось загрузить ни одного изображения")
        return None

    print(f"\nЗагружено изображений: {len(images_data)}")

    processor = ParallelImageProcessor(output_dir=str(run_output_dir))
    results = processor.process_images(images_data, method_name=method_name)

    print("Результаты обработки:")
    for index, object_id, path in results:
        print(f"  {index}. ID: {object_id} -> {Path(path).name}")

    print(f"\nВсего обработано: {len(results)} изображений")
    print(f"Результаты сохранены в: {run_output_dir}")



if __name__ == "__main__":
    asyncio.run(main())