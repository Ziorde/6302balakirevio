"""
Главный файл для демонстрации работы ООП версии лабораторной работы
"""

from processors.image_processor import ImageProcessor


def polymorphism(processor: ImageProcessor) -> None:
    """
    Демонстрация полиморфизма - обработка одного изображения
    разными способами (цветное и ч/б)

    Args:
        processor: Процессор с загруженным цветным изображением
    """
    grayscale_artwork = processor.convert_to_grayscale_artwork()

    if grayscale_artwork is None:
        return

    gs_processor = ImageProcessor()
    gs_processor._artwork = grayscale_artwork

    print("\nОбработка цветного изображения")
    color_sobel = processor.process_gaussian_blur(save=True)
    print("\nОбработка чёрно-белого изображения")
    gray_sobel = gs_processor.process_gaussian_blur(save=True)


def main() -> None:
    """
    Основная функция для демонстрации работы
    """
    processor = ImageProcessor()

    success = processor.load_from_met('MetObjects.csv')

    processor.show_info()

    processor.process_all()

    polymorphism(processor)

    processor2 = ImageProcessor()
    processor2.load_from_met('MetObjects.csv')

    processor2.combine_with(processor, save=True)



if __name__ == "__main__":
    main()