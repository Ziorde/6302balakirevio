from dataclasses import dataclass
from typing import Optional

@dataclass
class Metadata:
    """Класс для хранения метаданных произведения искусства"""
    object_id: str
    title: str
    artist_name: Optional[str] = None
    date: Optional[str] = None
    medium: Optional[str] = None
    dimensions: Optional[str] = None

    def __str__(self) -> str:
        return (f"Изображение: {self.title} (ID: {self.object_id})")
