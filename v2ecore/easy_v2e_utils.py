"""
Utils used in easy

@author: Piotr Baryczkowski
@contact: piotr.baryczkowski@student.put.poznan.pl
"""
import enum
import logging


class DVSParams(enum.Enum):
    """Easy optional setting of parameters for DVS model

    NONE:
        to use custom DVS paramters use this option.
    CLEAN:
        turns off noise, sets unlimited bandwidth and makes
        threshold variation small.
    NOISY:
        sets limited bandwidth and adds leak events and shot noise.
    """

    NONE = "none"
    NOISY = "noisy"
    CLEAN = "clean"

    def __str__(self):
        return str(self.value)


class DVSModel(enum.Enum):
    """DVS camera sizes.

    DVS 128: 128x128
    DVS 240: 240x180
    DVS 346: 346x260
    DVS 640: 640x480
    DVS 1024: 1024x768
    """

    DVS128 = (128, 128)
    DVS240 = (240, 180)
    DVS346 = (346, 260)
    DVS640 = (640, 480)
    DVS1024 = (1024, 768)

    def __str__(self):
        return str(self.value)


class DVSEventOutput(enum.Enum):
    """Output formats for DVS events.

    DDD_OUTPUT:
        Save frames, frame timestamp and corresponding event index
        in HDF5 format used for DDD17 and DDD20 datasets.
    DVS_H5:
        Output DVS events as hdf5 event database
    DVS_AEDAT2:
        Output DVS events as DAVIS346 camera AEDAT-2.0 event file for jAER.
        One file for real and one file for v2e events.
    DVS_AEDAT4:
        Output DV AEDAT-4.0 event file.
    DVS_TEXT:
        Output DVS events as text file with one event per
        line [timestamp (float s), x, y, polarity (0,1)].
    DVS_LAVA:
        Output DVS events as binary file with one event per
        line [x, y, polarity (0,1), timestamp (int ms)].
    """

    DDD_OUTPUT = 0
    DVS_H5 = ".h5"
    DVS_AEDAT2 = ".aedat"
    DVS_AEDAT4 = ".aedat4"
    DVS_TEXT = ".txt"
    DVS_LAVA = ".bin"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    def __str__(self):
        return str(self.value)


def easy_set_output_dimension(
    output_width: int, output_height: int, dvs_model: DVSModel, logger: logging.Logger
) -> tuple:
    """Return output_height and output_width based on arguments."""

    if dvs_model is not None:
        return dvs_model.value

    if (output_width is None) or (output_height is None):
        logger.warning(
            "Either output_width is None or output_height is None,"
            "or both. Setting both of them to None. \n"
            "Dimension will be set automatically from video input if available. \n"
            "Check DVS camera size arguments."
        )
        output_width, output_height = None, None

    return output_width, output_height
