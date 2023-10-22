"""
Python code for testing easy v2e.

@author: Piotr Baryczkowski
@contact: piotr.baryczkowski@student.put.poznan.pl
"""
import argparse
import logging
import sys

import torch

from v2ecore.easy_v2e import EasyV2EConverter
from v2ecore.easy_v2e_utils import DVSModel, DVSEventOutput


def parse_args(argv: list[str]) -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        "-i",
        "--input",
        type=str,
        action="store",
        required=True,
        help="Path to video.",
    )

    return arg_parser.parse_args()


def setup_logger() -> logging.Logger:
    logging.basicConfig()
    root = logging.getLogger()
    LOGGING_LEVEL = logging.INFO
    root.setLevel(LOGGING_LEVEL)  # todo move to info for production
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
    logging.addLevelName(
        logging.DEBUG, "\033[1;36m%s\033[1;0m" % logging.getLevelName(logging.DEBUG)
    )  # cyan foreground
    logging.addLevelName(
        logging.INFO, "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.INFO)
    )  # blue foreground
    logging.addLevelName(
        logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING)
    )  # red foreground
    logging.addLevelName(
        logging.ERROR, "\033[38;5;9m%s\033[1;0m" % logging.getLevelName(logging.ERROR)
    )  # red background
    logger = logging.getLogger(__name__)
    return logger


def main() -> None:
    args = parse_args(sys.argv[1:])
    logger = setup_logger()

    torch_device: str = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ).type
    logger.info(f"torch device is {torch_device}")
    if torch_device == "cpu":
        logger.warning(
            "CUDA GPU acceleration of pytorch operations is not available; "
            "see https://pytorch.org/get-started/locally/ "
            "to generate the correct conda install command to enable GPU-accelerated CUDA."
        )

    converter = EasyV2EConverter(
        torch_device,
        timestamp_resolution=0.003,
        auto_timestamp_resolution=False,
        pos_thres=0.15,
        neg_thres=0.15,
        sigma_thres=0.03,
        dvs_event_output=DVSEventOutput.DVS_AEDAT4,
        cutoff_hz=15,
        dvs_model=DVSModel.DVS346,
        slomo_model="/home/piotr/easy-v2e/input/SuperSloMo39.ckpt",
        logger=logger,
    )

    converter.convert_video(args.input, output_folder="output", dvs_vid="tennis")
    return


if __name__ == "__main__":
    main()
