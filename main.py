"""
Файл точка входа
"""

import config

import custom_image_processing as cip

import image_processing as ip

from met_api import download_painting

from utils import jpg_to_array, measure_time_and_save


def main() -> None:
    """
    Основная функция для обработки командной строки и выполнения обработки
    изображений.
    """

    painting = download_painting('MetObjects.csv')

    image = jpg_to_array(f'paintings/painting_{painting['object_id']}.jpg')

    # image = jpg_to_array('paintings/painting_45366.jpg')

    measure_time_and_save(cip.rgb_to_grayscale, image, 'my_rgb_to_grayscale.jpg')
    measure_time_and_save(ip.rgb_to_grayscale, image, 'rgb_to_grayscale.jpg')

    measure_time_and_save(cip.convolution, image, 'my_convolution.jpg',
                          config.SHARPNESS)
    measure_time_and_save(ip.convolution, image, 'convolution.jpg',
                          config.SHARPNESS)

    measure_time_and_save(cip.gaussian_blur, image, 'my_gaussian_blur.jpg')
    measure_time_and_save(ip.gaussian_blur, image, 'gaussian_blur.jpg')

    measure_time_and_save(cip.sobel_edge_detection, image, 'my_sobel_edge_detection.jpg')
    measure_time_and_save(ip.sobel_edge_detection, image, 'sobel_edge_detection.jpg')

    measure_time_and_save(cip.gamma_correction, image, 'my_gamma_correction.jpg')
    measure_time_and_save(ip.gamma_correction, image, 'gamma_correction.jpg')

    measure_time_and_save(cip.histogram_equalization, image, 'my_lab_equalization.jpg')
    measure_time_and_save(ip.histogram_equalization, image, 'lab_equalization.jpg')


if __name__ == "__main__":
    main()
