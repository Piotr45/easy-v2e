"""
Python code for visualising evs videos saved in AEDAT4 format.

@author: Piotr Baryczkowski
@contact: piotr.baryczkowski@student.put.poznan.pl
"""
import argparse
import sys

import aedat  # https://pypi.org/project/aedat/
import cv2  # https://pypi.org/project/opencv-python/
import numpy as np


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
        help="Path to file with AEDAT4 format",
    )

    arg_parser.add_argument(
        "--width",
        type=int,
        action="store",
        required=True,
        help="Width of output DVS data in pixels.",
    )

    arg_parser.add_argument(
        "--height",
        type=int,
        action="store",
        required=True,
        help="Height of output DVS data in pixels.",
    )

    return arg_parser.parse_args()


def visualise(decoder: aedat.Decoder, resolution: tuple[int, int]) -> None:
    for packet in decoder:
        image = np.full(resolution, 128, dtype=np.uint8)
        for event in packet["events"]:
            _, w, h, val = event
            if val != 0 and val != 1:
                print(val)

            pixel_color = 255 if val else 0
            if image[h, w] != pixel_color:
                image[h, w] = pixel_color

        cv2.imshow("event", image)
        if cv2.waitKey(33) == ord("q"):
            cv2.destroyAllWindows()
            sys.exit()


def main():
    args = parse_args(sys.argv[1:])

    decoder = aedat.Decoder(args.input)
    resolution = (args.height, args.width)

    visualise(decoder, resolution)


if __name__ == "__main__":
    main()
