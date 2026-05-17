"""
Модуль конфигурации логирования для проекта metetl.
Загружает конфигурацию из JSON файла.
"""

import json
import logging
import logging.config
from pathlib import Path


def setup_logging(
    config_path: str = None,
    console_level: int = None,
    file_level: int = None
) -> None:
    """
    Настройка логирования для проекта из JSON конфигурации.

    Args:
        config_path: Путь к JSON файлу конфигурации
        console_level: Уровень логирования для консоли (переопределяет конфиг)
        file_level: Уровень логирования для файла (переопределяет конфиг)
    """
    if config_path is None:
        config_path = Path(__file__).parent / "logging_config.json"

    if not Path(config_path).exists():
        _setup_default_logging()
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if console_level is not None:
        config['handlers']['console_handler']['level'] = console_level

    if file_level is not None:
        config['handlers']['file_handler']['level'] = file_level

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = config['handlers']['file_handler']['filename']
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logging.config.dictConfig(config)

    logger = logging.getLogger("metetl")
    logger.debug(
        f"Логирование настроено из {config_path}. "
        f"Файл: {log_file}, "
        f"уровень консоли: {config['handlers']['console_handler']['level']}, "
        f"уровень файла: {config['handlers']['file_handler']['level']}"
    )


def _setup_default_logging() -> None:
    """
    Настройка логирования по умолчанию (если JSON файл не найден).
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("metetl")
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

    file_handler = logging.FileHandler("logs/metetl.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.debug(
        "Логирование настроено (значения по умолчанию). "
        "Файл: logs/metetl.log, "
        "уровень консоли: INFO, уровень файла: DEBUG"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Получение логгера с указанным именем.

    Args:
        name: Имя логгера

    Returns:
        logging.Logger: Настроенный логгер
    """
    return logging.getLogger(f"metetl.{name}")