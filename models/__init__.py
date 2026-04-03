"""
Модуль с моделями данных для обработки изображений
"""

from .artwork import Artwork
from .metadata import Metadata
from .color_artwork import ColorArtwork
from .grayscale_artwork import GrayscaleArtwork

__all__ = [
    'Artwork',
    'Metadata',
    'ColorArtwork',
    'GrayscaleArtwork'
]