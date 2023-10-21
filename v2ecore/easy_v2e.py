import glob
import logging
import importlib
import os
import sys
from tempfile import TemporaryDirectory

import argcomplete
import cv2
import numpy as np
import torch
from engineering_notation import EngNumber as eng  # only from pip
from tqdm import tqdm

import v2ecore.desktop as desktop
from v2ecore.base_synthetic_input import base_synthetic_input
from v2ecore.emulator import EventEmulator
from v2ecore.renderer import EventRenderer, ExposureMode
from v2ecore.slomo import SuperSloMo
from v2ecore.v2e_args import (
    NO_SLOWDOWN,
    SmartFormatter,
    v2e_args,
    v2e_check_dvs_exposure_args,
    write_args_info,
)
from v2ecore.v2e_utils import (
    ImageFolderReader,
    all_images,
    check_lowpass,
    inputVideoFileDialog,
    read_image,
    set_output_dimension,
    set_output_folder,
    v2e_quit,
)
from v2ecore.easy_v2e_utils import DVSParams, DVSModel, DVSEventOutput


class EasyV2EConverter:
    """
    Python class for extracting frames from video file and synthesizing fake DVS
    events from this video after SuperSloMo has generated interpolated
    frames from the original video frames. This class is a boilerplate of original
    v2e.py file.

    @author: Piotr Baryczkowski
    @contact: piotr.baryczkowski@student.put.poznan.pl
    """

    def __init__(
        self,
        device: str,
        avi_frame_rate: int = 30,
        auto_timestamp_resolution: bool = True,
        timestamp_resolution: float | None = None,
        dvs_params: DVSParams = DVSParams.NONE,
        pos_thres: float = 0.2,
        neg_thres: float = 0.2,
        sigma_thres: float = 0.03,
        cutoff_hz: float = 300,
        leak_rate_hz: float = 0.01,
        shot_noise_rate_hz: float = 0.001,
        photoreceptor_noise: bool = True,
        leak_jitter_fraction: float = 0.1,
        noise_rate_cov_decades: float = 0.1,
        refractory_period: float = 0.0005,
        show_dvs_model_state: list[str] | None = None,
        save_dvs_model_state: bool = False,
        record_single_pixel_states: tuple | None = None,
        dvs_emulator_seed: int = 0,
        output_width: int = None,
        output_height: int = None,
        dvs_model: DVSModel | None = None,
        disable_slowmo: bool = False,
        slomo_model: str | None = None,
        batch_size: int = 8,
        hdr: bool = False,
        cs_lambda_pixels: float | None = None,
        cs_tau_p_ms: float | None = None,
        scidvs: bool = False,
        dvs_event_output: DVSEventOutput | None = DVSEventOutput.DVS_AEDAT4,
        label_signal_noise: str | None = None,
        dvs_exposure: list[str] = ("duration", "0.005"),
        dvs_vid_full_scale: int = 2,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Parameters
        ----------
        device:
            device, either 'cpu' or 'cuda' (selected automatically by caller
            depending on GPU availability)
        avi_frame_rate:
            TODO
        auto_timestamp_resolution:
            TODO
        timestamp_resolution:
            TODO
        dvs_params:
            TODO
        pos_thres:
            nominal threshold of triggering positive event in log intensity.
        neg_thres:
            nominal threshold of triggering negative event in log intensity.
        sigma_thres:
            std deviation of threshold in log intensity.
        cutoff_hz:
            3dB cutoff frequency in Hz of DVS photoreceptor
        leak_rate_hz:
            leak event rate per pixel in Hz,
            from junction leakage in reset switch
        shot_noise_rate_hz:
            shot noise rate in Hz
        photoreceptor_noise:
            model photoreceptor noise to create the desired shot noise rate
        leak_jitter_fraction:
            TODO
        noise_rate_cov_decades:
            TODO
        refractory_period:
            TODO
        show_dvs_model_state:
            None or 'new_frame','diff_frame' etc; see EventEmulator.MODEL_STATES
        save_dvs_model_state:
            TODO
        record_single_pixel_states:
            Record this pixel states to 'pixel_states.npy'
        dvs_emulator_seed:
            seed for random threshold variations,
            fix it to nonzero value to get same mismatch every time
        output_width: int,
            width of output in pixels
        output_height: int,
            height of output in pixels
        dvs_model:
            TODO
        disable_slowmo:
            TODO
        slomo_model:
            TODO
        batch_size:
            TODO
        hdr: bool
            Treat input as HDR floating point logarithmic
            gray scale with 255 input scaled as ln(255)=5.5441
        cs_lambda_pixels:
            space constant of surround in pixels, or None to disable surround inhibition
        cs_tau_p_ms:
            time constant of lowpass filter of surround in ms or 0 to make surround 'instantaneous'
        scidvs:
            Simulate the high gain adaptive photoreceptor SCIDVS pixel
        dvs_event_output
            names of output data files or None
        label_signal_noise: bool
            Record signal and noise event labels to a CSV file
        dvs_exposure:
            TODO
        dvs_vid_full_scale:
            TODO
        logger:
            TODO
        """
        self.device: str = device
        # timestamp resolution
        self._avi_frame_rate: int = avi_frame_rate
        self._auto_timestamp_resolution: bool = auto_timestamp_resolution
        self._timestamp_resolution: float = timestamp_resolution
        # DVS model parameters
        self._dvs_params: DVSParams = dvs_params  # NONE, NOISY, CLEAN
        self._pos_thres: float = pos_thres
        self._neg_thres: float = neg_thres
        self._sigma_thres: float = sigma_thres
        self._cutoff_hz: float = cutoff_hz
        self._leak_rate_hz: float = leak_rate_hz
        self._shot_noise_rate_hz: float = shot_noise_rate_hz
        self._photoreceptor_noise: bool = photoreceptor_noise
        self._leak_jitter_fraction: float = leak_jitter_fraction
        self._dvs_emulator_seed: int = dvs_emulator_seed
        self._noise_rate_cov_decades: float = noise_rate_cov_decades
        self._refractory_period: float = refractory_period
        self._dvs_emulator_seed: int = dvs_emulator_seed
        self._show_dvs_model_state: list[str] = show_dvs_model_state
        self._save_dvs_model_state: bool = save_dvs_model_state
        self._record_single_pixel_states: tuple | None = record_single_pixel_states
        # common camera types
        self._output_height: int = output_height
        self._output_width: int = output_width
        self._dvs_model: DVSModel = dvs_model  # DVS128 DVS240 DVS346 DVS640 DVS1024
        # slow motion frame synthesis
        self._disable_slowmo: bool = disable_slowmo
        self._slomo_model: str | None = slomo_model
        self._batch_size: int = batch_size
        self._hdr: bool = hdr
        # center surround DVS emulation
        self._cs_lambda_pixels: float = cs_lambda_pixels
        self._cs_tau_p_ms: float = cs_tau_p_ms
        # SCIDVS pixel study
        self._scvids: bool = scidvs
        # logging
        self._logger: logging.Logger = logger
        # output
        self._dvs_event_output: DVSEventOutput = dvs_event_output
        self._label_signal_noise: str | None = label_signal_noise  # TODO implement
        self._dvs_exposure: list[str] = dvs_exposure
        self._dvs_vid_full_scale: int = dvs_vid_full_scale
        # create emulator
        self._emulator: EventEmulator = self._create_emulator()

        # validate params
        self._validate_leak_rate()

    def convert_video(
        self, input_file: str, output_folder: str, dvs_vid: str, preview: bool = False
    ) -> None:
        """TODO"""
        self._validate_slomo(input_file)

        exposure_mode, exposure_val, area_dimension = self._v2e_check_dvs_exposure()

        if exposure_mode == ExposureMode.DURATION:
            dvsFps = 1.0 / exposure_val

        eventRenderer = EventRenderer(
            output_path=output_folder,
            dvs_vid=dvs_vid,
            preview=preview,
            full_scale_count=self._dvs_vid_full_scale,
            exposure_mode=exposure_mode,
            exposure_value=exposure_val,
            area_dimension=area_dimension,
            avi_frame_rate=self._avi_frame_rate,
        )
        return

    def _create_emulator(self) -> EventEmulator:
        """TODO"""
        (
            ddd_output,
            dvs_h5,
            dvs_aedat2,
            dvs_aedat4,
            dvs_text,
        ) = self._validate_dvs_output()
        emulator = EventEmulator(
            pos_thres=self._pos_thres,
            neg_thres=self._neg_thres,
            sigma_thres=self._sigma_thres,
            cutoff_hz=self._cutoff_hz,
            leak_rate_hz=self._leak_rate_hz,
            shot_noise_rate_hz=self._shot_noise_rate_hz,
            photoreceptor_noise=self._photoreceptor_noise,
            leak_jitter_fraction=self._leak_jitter_fraction,
            noise_rate_cov_decades=self._noise_rate_cov_decades,
            refractory_period_s=self._refractory_period,
            seed=self._dvs_emulator_seed,
            output_folder=None,
            dvs_h5=dvs_h5,
            dvs_aedat2=dvs_aedat2,
            dvs_aedat4=dvs_aedat4,
            dvs_text=dvs_text,
            show_dvs_model_state=self._show_dvs_model_state,
            save_dvs_model_state=self._save_dvs_model_state,
            output_width=self._output_width,
            output_height=self._output_height,
            device=self.device,
            cs_lambda_pixels=self._cs_lambda_pixels,
            cs_tau_p_ms=self._cs_tau_p_ms,
            hdr=self._hdr,
            scidvs=self._scvids,
            record_single_pixel_states=self._record_single_pixel_states,
            label_signal_noise=self._label_signal_noise,
        )

        if self._dvs_params is not None:
            if self._logger:
                self._logger.warning(
                    f"dvs_param={self._dvs_params} option overrides your "
                    f"selected options for threshold, threshold-mismatch, "
                    f"leak and shot noise rates"
                )

            emulator.set_dvs_params(self._dvs_params)

        return emulator

    def _validate_slomo(
        self, input_file: str, input_frame_rate: float | None = None
    ) -> None:
        """TODO"""
        cap = cv2.VideoCapture(input_file)
        srcFps = cap.get(cv2.CAP_PROP_FPS)
        srcNumFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if input_frame_rate is not None:
            if self._logger:
                self._logger.info(
                    f"Input video frame rate {srcFps}Hz is overridden by parsing input_frame_rate={input_frame_rate}."
                )
            srcFps = input_frame_rate

        if cap is not None:
            set_size = False
            if self._output_height is None and hasattr(cap, "frame_height"):
                set_size = True
                print(cap.frame_height)
                self._output_height = cap.frame_height
            if self._output_width is None and hasattr(cap, "frame_width"):
                set_size = True
                self._output_width = cap.frame_width
            if set_size and self._logger:
                self._logger.warning(
                    f"From input frame automatically set DVS output_width={self._output_width}"
                    f"and/or output_height={self._output_height}. "
                    f"This may not be desired behavior. \nCheck DVS camera sizes arguments."
                )
            elif (
                self._output_height is None
                or self._output_width is None
                and self._logger
            ):
                self._logger.warning(
                    "Could not read video frame size from video input and so could not automatically set DVS output size. \nCheck DVS camera sizes arguments."
                )

        # Check frame rate and number of frames
        if srcFps == 0:
            if self._logger:
                self._logger.error(
                    "source {} fps is 0; v2e needs to have a timescale "
                    "for input video".format(input_file)
                )
            v2e_quit()

        if srcNumFrames < 2:
            if self._logger:
                self._logger.warning(
                    "num frames is less than 2, probably cannot be determined "
                    "from cv2.CAP_PROP_FRAME_COUNT"
                )
            v2e_quit()

        return

    def _validate_dvs_output(self) -> list:
        """TODO"""
        return [
            dvs_out if dvs_out == self._dvs_event_output else None
            for dvs_out in DVSEventOutput.list()
        ]

    def _validate_leak_rate(self) -> None:
        if self._leak_rate_hz > 0 and self._sigma_thres == 0 and self._logger:
            self._logger.warning(
                "leak_rate_hz>0 but sigma_thres==0, "
                "so all leak events will be synchronous"
            )

    def _v2e_check_dvs_exposure(self) -> list:
        dvs_exposure = self._dvs_exposure
        exposure_mode = None
        exposure_val = None
        area_dimension = None
        try:
            exposure_mode = ExposureMode[dvs_exposure[0].upper()]
        except Exception:
            raise ValueError(
                "dvs_exposure first parameter '{}' must be 'duration','count', "
                " 'area_count' or 'source'".format(dvs_exposure[0])
            )

        if exposure_mode == ExposureMode.SOURCE:
            if self._logger:
                self._logger.info("DVS video exposure mode is SOURCE")
            return exposure_mode, None, None
        if exposure_mode == ExposureMode.AREA_COUNT and not len(dvs_exposure) == 3:
            raise ValueError(
                "area_event argument needs three parameters:  "
                "'area_count M N'; frame ends when any area of M x M pixels "
                "fills with N events"
            )
        elif (
            not exposure_mode == ExposureMode.AREA_COUNT and not len(dvs_exposure) == 2
        ):
            raise ValueError(
                "duration or count argument needs two parameters, "
                "e.g. 'duration 0.01' or 'count 3000'"
            )

        if not exposure_mode == ExposureMode.AREA_COUNT:
            try:
                exposure_val = float(dvs_exposure[1])
            except Exception:
                raise ValueError(
                    "dvs_exposure second parameter must be a number, "
                    "either duration or event count"
                )
        else:
            try:
                exposure_val = int(dvs_exposure[1])
                area_dimension = int(dvs_exposure[2])
            except Exception:
                raise ValueError(
                    "area_count must be N M, where N is event count and M "
                    "is area dimension in pixels"
                )
        s = "DVS frame expsosure mode {}".format(exposure_mode)
        if exposure_mode == ExposureMode.DURATION:
            s = s + ": frame rate {}".format(1.0 / exposure_val)
        elif exposure_mode == ExposureMode.COUNT:
            s = s + ": {} events/frame".format(exposure_val)
        elif exposure_mode == ExposureMode.AREA_COUNT:
            s = s + ": {} events per {}x{} pixel area".format(
                exposure_val, area_dimension, area_dimension
            )

        if self._logger:
            self._logger.info(s)
        return exposure_mode, exposure_val, area_dimension
