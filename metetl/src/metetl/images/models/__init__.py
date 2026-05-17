"""
Модели данных для изображений.
"""

from .metadata import Metadata
from .artwork import Artwork
from .color_artwork import ColorArtwork
from .grayscale_artwork import GrayscaleArtwork

__all__ = [
    "Metadata",
    "Artwork",
    "ColorArtwork",
    "GrayscaleArtwork",
]