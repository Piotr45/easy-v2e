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

    arg_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        action="store",
        required=False,
        default="output",
        help="Path to output directtory.",
    )

    arg_parser.add_argument(
        "--output-file",
        type=str,
        action="store",
        required=False,
        default="output",
        help="Filename of the output dvs video.",
    )

    arg_parser.add_argument(
        "--slomo-model",
        type=str,
        action="store",
        required=False,
        default=None,
        help="Path to SuperSloMo model.",
    )

    return arg_parser.parse_args()


def main() -> None:
    args = parse_args(sys.argv[1:])

    torch_device: str = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ).type

    converter = EasyV2EConverter(
        torch_device,
        timestamp_resolution=0.003,
        auto_timestamp_resolution=False,
        cutoff_hz=15,
        dvs_model=DVSModel.DVS346,
        slomo_model=args.slomo_model,
    )

    converter.convert_video(
        args.input,
        output_folder=args.output_dir,
        dvs_vid=args.output_file,
    )
    return


if __name__ == "__main__":
    main()
