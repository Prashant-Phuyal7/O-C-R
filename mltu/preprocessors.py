import os
import typing
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging

from . import Image
from mltu.annotations.audio import Audio

""" Implemented Preprocessors:
- ImageReader - Read image from path and return image and label
- ImageCropper - Crop image to (width, height)
"""

class ImageReader:
    """Read image from path and return image and label"""
    def __init__(self, image_class: Image, log_level: int = logging.INFO, ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self._image_class = image_class

    def __call__(self, image_path: typing.Union[str, np.ndarray], label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Read image from path and return image and label
        
        Args:
            image_path (typing.Union[str, np.ndarray]): Path to image or numpy array
            label (Any): Label of image

        Returns:
            Image: Image object
            Any: Label of image
        """
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found.")
        elif isinstance(image_path, np.ndarray):
            pass
        else:
            raise TypeError(f"Image {image_path} is not a string or numpy array.")

        image = self._image_class(image=image_path)

        if not image.init_successful:
            image = None
            self.logger.warning(f"Image {image_path} could not be read, returning None.")

        return image, label


class ImageCropper:
    """Crop image to (width, height)

    Attributes:
        width (int): Width of image
        height (int): Height of image
        wifth_offset (int): Offset for width
        height_offset (int): Offset for height
    """
    def __init__(
            self,
            width: int,
            height: int,
            width_offset: int = 0,
            height_offset: int = 0,
            log_level: int = logging.INFO
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self._width = width
        self._height = height
        self._width_offset = width_offset
        self._height_offset = height_offset

    def __call__(self, image: Image, label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        image_numpy = image.numpy()

        source_width, source_height = image_numpy.shape[:2][::-1]

        if source_width >= self._width:
            image_numpy = image_numpy[:, self._width_offset:self._width + self._width_offset]
        else:
            raise Exception("unexpected")

        if source_height >= self._height:
            image_numpy = image_numpy[self._height_offset:self._height + self._height_offset, :]
        else:
            raise Exception("unexpected")

        image.update(image_numpy)

        return image, label
