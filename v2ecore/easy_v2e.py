import glob
import logging
import importlib
import os
import sys
from tempfile import TemporaryDirectory
from typing import Any

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
from v2ecore.easy_v2e_utils import (
    DVSParams,
    DVSModel,
    DVSEventOutput,
    easy_set_output_dimension,
)


class EasyV2EConverter:
    """
    Python class for extracting frames from video file and synthesizing fake DVS
    events from this video after SuperSloMo has generated interpolated
    frames from the original video frames. This class is a boilerplate of original
    v2e.py file.

    @author: Piotr Baryczkowski
    @contact: piotr.baryczkowski@student.put.poznan.pl
    """

    @staticmethod
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
            logging.WARNING,
            "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING),
        )  # red foreground
        logging.addLevelName(
            logging.ERROR,
            "\033[38;5;9m%s\033[1;0m" % logging.getLevelName(logging.ERROR),
        )  # red background
        logger = logging.getLogger(__name__)
        return logger

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
        disable_slomo: bool = False,
        input_slowmotion_factor: float = 1.0,
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
        disable_logger: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        device:
            Device, either 'cpu' or 'cuda' (selected automatically by caller
            depending on GPU availability)
        avi_frame_rate:
            Frame rate of output AVI video files; only affects playback rate.
        auto_timestamp_resolution:
            (Ignored by disable_slomo or synthetic_input.) If
            True (default), upsampling_factor is automatically
            determined to limit maximum movement between frames to
            1 pixel. If False, timestamp_resolution sets the
            upsampling factor for input video. Can be combined
            with timestamp_resolution to ensure DVS events have
            at most some resolution.
        timestamp_resolution:
            (Ignored by disable_slomo or synthetic_input.)
            Desired DVS timestamp resolution in seconds;
            determines slow motion upsampling factor; the video
            will be upsampled from source fps to achieve the at
            least this timestamp resolution.I.e. slowdown_factor =
            (1/fps)/timestamp_resolution; using a high resolution
            e.g. of 1ms will result in slow rendering since it
            will force high upsampling ratio. Can be combind with
            auto_timestamp_resolution to limit upsampling to a
            maximum limit value.
        dvs_params:
            Easy optional setting of parameters for DVS
            model:None, 'clean', 'noisy'; 'clean' turns off noise,
            sets unlimited bandwidth and makes threshold variation
            small. 'noisy' sets limited bandwidth and adds leak
            events and shot noise.This option by default will
            disable user set DVS parameters. To use custom DVS
            paramters, use None here.
        pos_thres:
            Nominal threshold of triggering positive event in log intensity.
        neg_thres:
            Nominal threshold of triggering negative event in log intensity.
        sigma_thres:
            Std deviation of threshold in log intensity.
        cutoff_hz:
            3dB cutoff frequency in Hz of DVS photoreceptor
        leak_rate_hz:
            Leak event rate per pixel in Hz,
            from junction leakage in reset switch
        shot_noise_rate_hz:
            Shot noise rate in Hz
        photoreceptor_noise:
            Model photoreceptor noise to create the desired shot noise rate
        leak_jitter_fraction:
            Jitter of leak noise events relative to the (FPN)
            interval, drawn from normal distribution
        noise_rate_cov_decades:
            Coefficient of Variation of noise rates (shot and
            leak) in log normal distribution decades across pixel
            arrayWARNING: currently only in leak events
        refractory_period:
            Refractory period in seconds, default is 0.5ms.The new
            event will be ignore if the previous event is
            triggered less than refractory_period ago.Set to 0 to
            disable this feature.
        show_dvs_model_state:
            None or 'new_frame','diff_frame' etc; see EventEmulator.MODEL_STATES
        save_dvs_model_state:
            One or more space separated list model states. Possible
            models states are (without quotes) either 'all' or
            chosen from dict_keys(['new_frame', 'log_new_frame',
            'lp_log_frame', 'scidvs_highpass',
            'photoreceptor_noise_arr', 'cs_surround_frame',
            'c_minus_s_frame', 'base_log_frame', 'diff_frame'])
        record_single_pixel_states:
            Record this pixel states to 'pixel_states.npy'
        dvs_emulator_seed:
            Seed for random threshold variations,
            fix it to nonzero value to get same mismatch every time
        output_width: int,
            Width of output in pixels
        output_height: int,
            Height of output in pixels
        dvs_model:
            DVS camera model with example video resolution e.g. DVS128: 128x128
        disable_slomo:
            Disables slomo interpolation; the output DVS events
            will have exactly the timestamp resolution of the
            source video (which is perhaps modified by
            input_slowmotion_factor).
        slomo_model:
            Path of slomo_model checkpoint.
        batch_size:
            Batch size in frames for SuperSloMo. Batch size 8-16
            is recommended if your GPU has sufficient memory.
        hdr: bool
            Treat input as HDR floating point logarithmic
            gray scale with 255 input scaled as ln(255)=5.5441
        cs_lambda_pixels:
            Space constant of surround in pixels, or None to disable surround inhibition
        cs_tau_p_ms:
            Time constant of lowpass filter of surround in ms or 0 to make surround 'instantaneous'
        scidvs:
            Simulate the high gain adaptive photoreceptor SCIDVS pixel
        dvs_event_output
            Names of output data files or None
        label_signal_noise: bool
            Record signal and noise event labels to a CSV file
        dvs_exposure:
            Mode to finish DVS frame event integration:
                - duration time: Use fixed accumulation time in seconds, e.g.
                    dvs_exposure=("duration",".005");
                - count n: Count n events per frame,e.g.
                    dvs_exposure=("count","5000");
                - area_count M N: frame ends when any area of N x N pixels fills with M events, e.g.
                    dvs_exposure=("area_count","500","64")
                source: each DVS frame is from one source frame (slomo or original, depending on if slomo is used)
        dvs_vid_full_scale:
            Set full scale event count histogram count for DVS
            videos to be this many ON or OFF events for full white
            or black.
        logger:
            Logger class object to log progress of the module.
        disable_logger:
            Disables logger.
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
        self._disable_slomo: bool = disable_slomo
        self._input_slowmotion_factor: bool = input_slowmotion_factor
        self._slomo_model: str | None = slomo_model
        self._batch_size: int = batch_size
        self._hdr: bool = hdr
        # center surround DVS emulation
        self._cs_lambda_pixels: float = cs_lambda_pixels
        self._cs_tau_p_ms: float = cs_tau_p_ms
        # SCIDVS pixel study
        self._scvids: bool = scidvs
        # logging
        if logger:
            self._logger: logging.Logger = logger
        else:
            self._logger: logging.Logger = self.setup_logger()
        if disable_logger:
            logging.disable()
        # output
        self._dvs_event_output: DVSEventOutput = dvs_event_output
        self._label_signal_noise: str | None = label_signal_noise  # TODO implement
        self._dvs_exposure: list[str] = dvs_exposure
        self._dvs_vid_full_scale: int = dvs_vid_full_scale
        # set output width and height based on the arguments
        self._output_width, self._output_height = easy_set_output_dimension(
            self._output_width, self._output_height, self._dvs_model, self._logger
        )
        # validate params
        self._validate_video_parameters()
        self._validate_leak_rate()
        # DVS
        self._emulator: EventEmulator | None = None
        self._event_renderer: EventRenderer | None = None
        self._slomo: SuperSloMo | None = None
        # TODO temp
        self._slowdown_factor = None

    def convert_video(
        self,
        input_file: str,
        output_folder: str,
        dvs_vid: str | None = None,
        input_start_time: float | None = None,
        input_stop_time: float | None = None,
        overwrite: bool = False,
        preview: bool = False,
    ) -> None:
        """Converts video into event stream

        Parameters
        ----------
        input_file: str
            Path to input file, that will be converted.
        output_folder: str
            Path to output directory.
        dvs_vid: str
            TODO
        input_start_time: float
            TODO
        input_stop_time: float
            TODO
        overwrite: bool
            TODO
        preview: bool
            TODO
        """
        # input file checking
        self._validate_input(input_file)
        # TODO change to params
        self._check_input_time(input_start_time, input_stop_time)

        exposure_mode, exposure_val, area_dimension = self._easy_check_dvs_exposure()

        if exposure_mode == ExposureMode.DURATION:
            dvs_fps = 1.0 / exposure_val

        cap, dvs_fps, src_num_frames = self._validate_input_file_type(input_file)
        (
            src_total_duration,
            src_frame_interval_s,
            start_frame,
            stop_frame,
            src_num_frames_to_be_proccessed,
            start_time,
            stop_time,
        ) = self._validate_slomo(
            dvs_fps, src_num_frames, input_start_time, input_stop_time
        )

        # create emulator
        self._emulator: EventEmulator = self._create_emulator(dvs_vid, output_folder)

        self._event_renderer = EventRenderer(
            output_path=output_folder,
            dvs_vid=dvs_vid,
            preview=preview,
            full_scale_count=self._dvs_vid_full_scale,
            exposure_mode=exposure_mode,
            exposure_value=exposure_val,
            area_dimension=area_dimension,
            avi_frame_rate=self._avi_frame_rate,
        )

        self._run_stages(
            cap,
            input_file,
            src_num_frames_to_be_proccessed,
            src_frame_interval_s,
            start_frame,
            stop_frame,
            start_time,
            stop_time,
        )

        self._clean_up()

        return

    def _create_emulator(self, output_file: str, output_folder: str) -> EventEmulator:
        """Function that creates emulator object."""
        (
            ddd_output,
            dvs_h5,
            dvs_aedat2,
            dvs_aedat4,
            dvs_text,
        ) = self._validate_dvs_output(output_file)
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
            output_folder=output_folder,
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
            self._logger.warning(
                f"dvs_param={self._dvs_params} option overrides your "
                f"selected options for threshold, threshold-mismatch, "
                f"leak and shot noise rates"
            )

            emulator.set_dvs_params(self._dvs_params)

        return emulator

    def _check_input_time(self, input_start_time, input_stop_time) -> tuple:
        """TODO"""

        def is_float(element: Any) -> bool:
            try:
                float(element)
                return True
            except ValueError:
                return False

        if (
            not input_start_time is None
            and not input_stop_time is None
            and is_float(input_start_time)
            and is_float(input_stop_time)
            and input_stop_time <= input_start_time
        ):
            self._logger.error(
                f"stop time {input_stop_time} must be later than start time {input_start_time}"
            )
            v2e_quit(1)

    def _validate_input_file_type(
        self, input_file: str, input_frame_rate: float | None = None
    ) -> tuple:
        """TODO"""
        if os.path.isdir(input_file):
            if input_frame_rate is None:
                self._logger.error(
                    "When the video is presented as a folder, "
                    "The user must set input_frame_rate manually"
                )
                v2e_quit(1)

            cap = ImageFolderReader(input_file, self._input_frame_rate)
            src_fps = cap.frame_rate
            src_num_frames = cap.num_frames
        else:
            cap = cv2.VideoCapture(input_file)
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            src_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if input_frame_rate is not None:
                self._logger.info(
                    f"Input video frame rate {src_fps}Hz is overridden by parsing input_frame_rate={input_frame_rate}."
                )
                src_fps = input_frame_rate

        if cap is not None:
            set_size = False
            if self._output_height is None and hasattr(cap, "frame_height"):
                set_size = True
                print(cap.frame_height)
                self._output_height = cap.frame_height
            if self._output_width is None and hasattr(cap, "frame_width"):
                set_size = True
                self._output_width = cap.frame_width
            if set_size:
                self._logger.warning(
                    f"From input frame automatically set DVS output_width={self._output_width}"
                    f"and/or output_height={self._output_height}. "
                    f"This may not be desired behavior. \nCheck DVS camera sizes arguments."
                )
            elif self._output_height is None or self._output_width is None:
                self._logger.warning(
                    "Could not read video frame size from video input and so could not automatically set DVS output size. \nCheck DVS camera sizes arguments."
                )

        # Check frame rate and number of frames
        if src_fps == 0:
            self._logger.error(
                "source {} fps is 0; v2e needs to have a timescale "
                "for input video".format(input_file)
            )
            v2e_quit()

        if src_num_frames < 2:
            self._logger.warning(
                "num frames is less than 2, probably cannot be determined "
                "from cv2.CAP_PROP_FRAME_COUNT"
            )
            v2e_quit()

        return cap, src_fps, src_num_frames

    def _validate_dvs_output(self, output_file: str) -> list:
        """TODO"""
        return [
            output_file if dvs_out == self._dvs_event_output.value else None
            for dvs_out in DVSEventOutput.list()
        ]

    def _validate_leak_rate(self) -> None:
        """TODO"""
        if self._leak_rate_hz > 0 and self._sigma_thres == 0:
            self._logger.warning(
                "leak_rate_hz>0 but sigma_thres==0, "
                "so all leak events will be synchronous"
            )

    def _validate_input(self, input_file: str) -> None:
        """TODO"""
        if not os.path.isfile(input_file) and not os.path.isdir(input_file):
            self._logger.error("input file {} does not exist".format(input_file))
            v2e_quit(1)
        if os.path.isdir(input_file):
            if len(os.listdir(input_file)) == 0:
                self._logger.error(f"input folder {input_file} is empty")
                v2e_quit(1)

    def _validate_video_parameters(self) -> None:
        """TODO"""
        if (
            not self._disable_slomo
            and self._auto_timestamp_resolution is False
            and self._timestamp_resolution is None
        ):
            self._logger.error(
                "if auto_timestamp_resolution=False, "
                "then timestamp_resolution must be set to "
                "some desired DVS event timestamp resolution in seconds, "
                "e.g. 0.01"
            )
            v2e_quit()

        if (
            self._auto_timestamp_resolution is True
            and self._timestamp_resolution is not None
        ):
            self._logger.info(
                f"auto_timestamp_resolution=True and "
                f"timestamp_resolution={self._timestamp_resolution}: "
                f"Limiting automatic upsampling to maximum timestamp interval."
            )

    def _validate_slomo(
        self,
        src_fps: float,
        src_num_frames: float,
        input_start_time: float | None = None,
        input_stop_time: float | None = None,
        preview: bool = False,
    ) -> tuple:
        """TODO"""
        src_total_duration = (src_num_frames - 1) / src_fps
        # the index of the frames, from 0 to srcNumFrames-1
        start_frame = (
            int(src_num_frames * (input_start_time / src_total_duration))
            if input_start_time
            else 0
        )
        stop_frame = (
            int(src_num_frames * (input_stop_time / src_total_duration))
            if input_stop_time
            else src_num_frames - 1
        )
        src_num_frames_to_be_proccessed = stop_frame - start_frame + 1
        # the duration to be processed, should subtract 1 frame when
        # calculating duration
        src_duration_to_be_processed = (src_num_frames_to_be_proccessed - 1) / src_fps

        # redefining start and end time using the time calculated
        # from the frames, the minimum resolution there is
        start_time = start_frame / src_fps
        stop_time = stop_frame / src_fps

        src_frame_interval_s = (1.0 / src_fps) / self._input_slowmotion_factor

        slowdown_factor = NO_SLOWDOWN
        if self._disable_slomo:
            self._logger.warning(
                "slomo interpolation disabled by command line option; "
                "output DVS timestamps will have source frame interval "
                "resolution"
            )
            # time stamp resolution equals to source frame interval
            slomo_timestamp_tesolution_s = src_frame_interval_s
        elif not self._auto_timestamp_resolution:
            slowdown_factor = int(
                np.ceil(src_frame_interval_s / self._timestamp_resolution)
            )
            if slowdown_factor < NO_SLOWDOWN:
                slowdown_factor = NO_SLOWDOWN
                self._logger.warning(
                    "timestamp resolution={}s is >= source "
                    "frame interval={}s, will not upsample".format(
                        self._timestamp_resolution, src_frame_interval_s
                    )
                )
            elif slowdown_factor > 100 and self._cutoff_hz == 0:
                self._logger.warning(
                    f"slowdown_factor={slowdown_factor} is >100 but "
                    "cutoff_hz={cutoff_hz}. We have observed that "
                    "numerical errors in SuperSloMo can cause noise "
                    "that makes fake events at the upsampling rate. "
                    "Recommend to set physical cutoff_hz, "
                    "e.g. --cutoff_hz=200 (or leave the default cutoff_hz)"
                )
            slomo_timestamp_tesolution_s = src_frame_interval_s / slowdown_factor

            self._logger.info(
                f"auto_timestamp_resolution is False, "
                f"srcFps={src_fps}Hz "
                f"input_slowmotion_factor={self._input_slowmotion_factor}, "
                f"real src FPS={src_fps*self._input_slowmotion_factor}Hz, "
                f"srcFrameIntervalS={eng(src_frame_interval_s)}s, "
                f"timestamp_resolution={eng(self._timestamp_resolution)}s, "
                f"so SuperSloMo will use slowdown_factor={slowdown_factor} "
                f"and have "
                f"slomoTimestampResolutionS={eng(slomo_timestamp_tesolution_s)}s"
            )

            if slomo_timestamp_tesolution_s > self._timestamp_resolution:
                self._logger.warning(
                    "Upsampled src frame intervals of {}s is larger than\n "
                    "the desired DVS timestamp resolution of {}s".format(
                        slomo_timestamp_tesolution_s, self._timestamp_resolution
                    )
                )

            check_lowpass(
                self._cutoff_hz, 1 / slomo_timestamp_tesolution_s, self._logger
            )
        else:  # auto_timestamp_resolution
            if self._timestamp_resolution is not None:
                self._slowdown_factor = int(
                    np.ceil(src_frame_interval_s / self._timestamp_resolution)
                )

                self._logger.info(
                    f"auto_timestamp_resolution=True and "
                    f"timestamp_resolution={eng(self._timestamp_resolution)}s: "
                    f"source video will be automatically upsampled but "
                    f"with at least upsampling factor of {slowdown_factor}"
                )
            else:
                self._logger.info(
                    "auto_timestamp_resolution=True and "
                    "timestamp_resolution is not set: "
                    "source video will be automatically upsampled to "
                    "limit maximum interframe motion to 1 pixel"
                )

        # the SloMo model, set no SloMo model if no slowdown
        if not self._disable_slomo and (
            self._auto_timestamp_resolution or slowdown_factor != NO_SLOWDOWN
        ):
            self._slomo = SuperSloMo(
                model=self._slomo_model,
                auto_upsample=self._auto_timestamp_resolution,
                upsampling_factor=slowdown_factor,
                video_path=None,  # TODO add params
                vid_orig=None,
                vid_slomo=None,
                preview=preview,
                batch_size=self._batch_size,
            )
            return (
                src_total_duration,
                src_frame_interval_s,
                start_frame,
                stop_frame,
                src_num_frames_to_be_proccessed,
                start_time,
                stop_time,
            )

        return (
            src_total_duration,
            src_frame_interval_s,
            start_frame,
            stop_frame,
            src_num_frames_to_be_proccessed,
            start_time,
            stop_time,
        )

    def _easy_check_dvs_exposure(self) -> list:
        """TODO"""
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

            self._logger.info(s)
        return exposure_mode, exposure_val, area_dimension

    def _run_stages(
        self,
        cap: cv2.VideoCapture,
        input_file: str,
        src_num_frames_to_be_proccessed: int,
        src_frame_interval_s,
        start_frame,
        stop_frame,
        start_time,
        stop_time,
    ) -> None:
        """TODO"""
        # timestamps of DVS start at zero and end with
        # span of video we processed
        src_video_real_processed_duration = (
            stop_time - start_time
        ) / self._input_slowmotion_factor
        num_frames = src_num_frames_to_be_proccessed
        inputHeight = None
        inputWidth = None
        inputChannels = None
        if start_frame > 0:
            self._logger.info("skipping to frame {}".format(start_frame))
            for i in tqdm(range(start_frame), unit="fr", desc="src"):
                if isinstance(cap, ImageFolderReader):
                    if i < start_frame - 1:
                        ret, _ = cap.read(skip=True)
                    else:
                        ret, _ = cap.read()
                else:
                    ret, _ = cap.read()
                if not ret:
                    raise ValueError(
                        "something wrong, got to end of file before "
                        "reaching start_frame"
                    )

        self._logger.info(
            "processing frames {} to {} from video input".format(
                start_frame, stop_frame
            )
        )
        num_frames = src_num_frames_to_be_proccessed

        with TemporaryDirectory() as source_frames_dir:
            if os.path.isdir(input_file):  # folder input
                inputWidth = cap.frame_width
                inputHeight = cap.frame_height
                inputChannels = cap.frame_channels
            else:
                inputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                inputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                inputChannels = 1 if int(cap.get(cv2.CAP_PROP_MONOCHROME)) else 3
            self._logger.info(
                "Input video {} has W={} x H={} frames each with {} channels".format(
                    input_file, inputWidth, inputHeight, inputChannels
                )
            )

            if (self._output_width is None) and (self._output_height is None):
                self._output_width = inputWidth
                self._output_height = inputHeight
                self._logger.warning(
                    "output size ({}x{}) was set automatically to "
                    "input video size\n    Are you sure you want this? "
                    "It might be slow.\n Consider using\n "
                    "    --output_width=346 --output_height=260\n "
                    "to match Davis346.".format(self._output_width, self._output_height)
                )

                # set emulator output width and height for the last time
                self._emulator.output_width = self._output_width
                self._emulator.output_height = self._output_height

                self._logger.info(
                    f"*** Stage 1/3: "
                    f"Resizing {src_num_frames_to_be_proccessed} input frames "
                    f"to output size "
                    f"(with possible RGB to luma conversion)"
                )
            for inputFrameIndex in tqdm(
                range(src_num_frames_to_be_proccessed), desc="rgb2luma", unit="fr"
            ):
                # read frame
                ret, input_video_frame = cap.read()
                num_frames += 1
                if ret == False:
                    self._logger.warning(
                        f"could not read frame {inputFrameIndex} from {cap}"
                    )
                    continue
                if input_video_frame is None or np.shape(input_video_frame) == ():
                    self._logger.warning(
                        f"empty video frame number {inputFrameIndex} in {cap}"
                    )
                    continue
                if not ret or inputFrameIndex + start_frame > stop_frame:
                    break

                if (
                    self._output_height
                    and self._output_width
                    and (
                        inputHeight != self._output_height
                        or inputWidth != self._output_width
                    )
                ):
                    dim = (self._output_width, self._output_height)
                    (fx, fy) = (
                        float(self._output_width) / inputWidth,
                        float(self._output_height) / inputHeight,
                    )
                    input_video_frame = cv2.resize(
                        src=input_video_frame,
                        dsize=dim,
                        fx=fx,
                        fy=fy,
                        interpolation=cv2.INTER_AREA,
                    )
                if inputChannels == 3:  # color
                    if inputFrameIndex == 0:  # print info once
                        self._logger.info(
                            "\nConverting input frames from RGB color to luma"
                        )
                    # TODO would break resize if input is gray frames
                    # convert RGB frame into luminance.
                    input_video_frame = cv2.cvtColor(
                        input_video_frame, cv2.COLOR_BGR2GRAY
                    )  # much faster

                    # TODO add vid_orig output if not using slomo

                # save frame into numpy records
                save_path = os.path.join(
                    source_frames_dir, str(inputFrameIndex).zfill(8) + ".npy"
                )
                np.save(save_path, input_video_frame)
                # print("Writing source frame {}".format(save_path), end="\r")
            cap.release()

            with TemporaryDirectory() as interpFramesFolder:
                interpTimes = None
                # make input to slomo
                if self._slomo is not None and (
                    self._auto_timestamp_resolution
                    or self._slowdown_factor != NO_SLOWDOWN
                ):
                    # interpolated frames are stored to tmpfolder as
                    # 1.png, 2.png, etc
                    self._logger.info(
                        f"*** Stage 2/3: SloMo upsampling from " f"{source_frames_dir}"
                    )
                    interpTimes, avgUpsamplingFactor = self._slomo.interpolate(
                        source_frames_dir,
                        interpFramesFolder,
                        (self._output_width, self._output_height),
                    )
                    avgTs = src_frame_interval_s / avgUpsamplingFactor
                    self._logger.info(
                        "SloMo average upsampling factor={:5.2f}; "
                        "average DVS timestamp resolution={}s".format(
                            avgUpsamplingFactor, eng(avgTs)
                        )
                    )
                    # check for undersampling wrt the
                    # photoreceptor lowpass filtering

                    if self._cutoff_hz > 0:
                        self._logger.info(
                            "Using auto_timestamp_resolution. "
                            "checking if cutoff hz is ok given "
                            "sample rate {}".format(1 / avgTs)
                        )
                        check_lowpass(self._cutoff_hz, 1 / avgTs, self._logger)

                    # read back to memory
                    interpFramesFilenames = all_images(interpFramesFolder)
                    # number of frames
                    n = len(interpFramesFilenames)
                else:
                    self._logger.info(
                        f"*** Stage 2/3:turning npy frame files to png "
                        f"from {source_frames_dir}"
                    )
                    interpFramesFilenames = []
                    n = 0
                    src_files = sorted(
                        glob.glob("{}".format(source_frames_dir) + "/*.npy")
                    )
                    for frame_idx, src_file_path in tqdm(
                        enumerate(src_files), desc="npy2png", unit="fr"
                    ):
                        src_frame = np.load(src_file_path)
                        tgt_file_path = os.path.join(
                            interpFramesFolder, str(frame_idx) + ".png"
                        )
                        interpFramesFilenames.append(tgt_file_path)
                        n += 1
                        cv2.imwrite(tgt_file_path, src_frame)
                    interpTimes = np.array(range(n))

                # compute times of output integrated frames
                nFrames = len(interpFramesFilenames)
                # interpTimes is in units of 1 per input frame,
                # normalize it to src video time range
                f = src_video_real_processed_duration / (
                    np.max(interpTimes) - np.min(interpTimes)
                )
                # compute actual times from video times
                interpTimes = f * interpTimes

                # array to batch events for rendering to DVS frames
                events = np.zeros((0, 4), dtype=np.float32)

                self._logger.info(
                    f"*** Stage 3/3: emulating DVS events from " f"{nFrames} frames"
                )

                # parepare extra steps for data storage
                # right before event emulation
                # if args.ddd_output:
                #     self._emulator.prepare_storage(nFrames, interpTimes)

                # generate events from frames and accumulate events to DVS frames for output DVS video
                with tqdm(total=nFrames, desc="dvs", unit="fr") as pbar:
                    with torch.no_grad():
                        for i in range(nFrames):
                            fr = read_image(interpFramesFilenames[i])
                            newEvents = self._emulator.generate_events(
                                fr, interpTimes[i]
                            )

                            pbar.update(1)
                            if newEvents is not None and newEvents.shape[0] > 0:
                                events = np.append(events, newEvents, axis=0)
                                events = np.array(events)
                                if i % self._batch_size == 0:
                                    self._event_renderer.render_events_to_frames(
                                        events,
                                        height=self._output_height,
                                        width=self._output_width,
                                    )
                                    events = np.zeros((0, 4), dtype=np.float32)
                    # process leftover events
                    if len(events) > 0:
                        self._event_renderer.render_events_to_frames(
                            events, height=self._output_height, width=self._output_width
                        )
        return

    def _clean_up(self) -> None:
        self._event_renderer.cleanup()
        self._emulator.cleanup()
        if self._slomo is not None:
            self._slomo.cleanup()
