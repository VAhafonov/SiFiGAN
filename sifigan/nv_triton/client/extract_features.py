# -*- coding: utf-8 -*-

"""Feature extraction script.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG
    - https://github.com/k2kobayashi/sprocket

"""
import argparse
import copy
import multiprocessing as mp
import os
from typing import List, Tuple

import librosa
import numpy as np
import pysptk
import pyworld
import soundfile as sf
import yaml
from scipy.interpolate import interp1d
from scipy.signal import firwin, lfilter
from sifigan.utils import read_txt, write_hdf5


# All-pass-filter coefficients {key -> sampling rate : value -> coefficient}
ALPHA = {
    8000: 0.312,
    12000: 0.369,
    16000: 0.410,
    22050: 0.455,
    24000: 0.466,
    32000: 0.504,
    44100: 0.544,
    48000: 0.554,
}

DEFAULT_HDF5_EXTENSION = '.h5'


def low_cut_filter(x: np.ndarray, fs: int, cutoff: float = 70) -> np.ndarray:
    """Low cut filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence

    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def low_pass_filter(x: np.ndarray, fs: int, cutoff: float = 70) -> np.ndarray:
    """Low pass filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence

    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), "edge")
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2 : -numtaps // 2]

    return lpf_x


def convert_continuos_f0(f0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Convert F0 to continuous F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)

    """
    # get uv information as binary
    uv = np.float32(f0 != 0)
    # get start and end of f0
    if (f0 == 0).all():
        print("WARINING: all of the f0 values are 0.")
        return uv, f0, False
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    # padding start and end of f0 sequence
    cf0 = copy.deepcopy(f0)
    start_idx = np.where(cf0 == start_f0)[0][0]
    end_idx = np.where(cf0 == end_f0)[0][-1]
    cf0[:start_idx] = start_f0
    cf0[end_idx:] = end_f0
    # get non-zero frame index
    nz_frames = np.where(cf0 != 0)[0]
    # perform linear interpolation
    f = interp1d(nz_frames, cf0[nz_frames])
    cf0 = f(np.arange(0, cf0.shape[0]))

    return uv, cf0, True


def world_feature_extract(queue, wav_list, config, out_dir):
    """WORLD feature extraction

    Args:
        queue (multiprocessing.Queue): the queue to store the file name of utterance
        wav_list (list): list of the wav files
        config (dict): feature extraction config
        out_dir (str): dir where to save extracted features

    """
    sample_rate = config['sample_rate']
    highpass_cutoff = config['highpass_cutoff']
    minf0 = config['minf0']
    maxf0 = config['maxf0']
    shiftms = config['shiftms']
    fft_size = config['fft_size']
    hop_size = config['hop_size']
    mcep_dim = config['mcep_dim']
    # extraction
    for i, wav_name in enumerate(wav_list):
        print(f"now processing {wav_name} ({i + 1}/{len(wav_list)})")

        # load wavfile
        x, fs = sf.read(wav_name)
        x = np.array(x, dtype=float)

        # check sampling frequency
        if not fs == sample_rate:
            print("WARNING:" +
                f"Sampling frequency of {wav_name} is not matched."
                + "Resample before feature extraction."
            )
            x = librosa.resample(x, orig_sr=fs, target_sr=sample_rate)

        # apply low-cut-filter
        if highpass_cutoff > 0:
            if (x == 0).all():
                print(f"xxxxx {wav_name}")
                continue
            x = low_cut_filter(x, sample_rate, cutoff=highpass_cutoff)

        # extract WORLD features
        f0, t = pyworld.harvest(
            x,
            fs=sample_rate,
            f0_floor=minf0,
            f0_ceil=maxf0,
            frame_period=shiftms,
        )
        env = pyworld.cheaptrick(
            x,
            f0,
            t,
            fs=sample_rate,
            fft_size=fft_size,
        )
        ap = pyworld.d4c(
            x,
            f0,
            t,
            fs=sample_rate,
            fft_size=fft_size,
        )
        uv, cf0, is_all_uv = convert_continuos_f0(f0)
        if is_all_uv:
            lpf_fs = int(sample_rate / hop_size)
            cf0_lpf = low_pass_filter(cf0, lpf_fs, cutoff=20)
            next_cutoff = 70
            while not (cf0_lpf >= [0]).all():
                cf0_lpf = low_pass_filter(cf0, lpf_fs, cutoff=next_cutoff)
                next_cutoff *= 2
        else:
            cf0_lpf = cf0
            print("WARNING: " + f"all of the f0 values are 0 {wav_name}.")
        mcep = pysptk.sp2mc(env, order=mcep_dim, alpha=ALPHA[sample_rate])
        bap = pyworld.code_aperiodicity(ap, sample_rate)

        # adjust shapes
        minlen = min(uv.shape[0], mcep.shape[0])
        uv = np.expand_dims(uv[:minlen], axis=-1)
        f0 = np.expand_dims(f0[:minlen], axis=-1)
        cf0_lpf = np.expand_dims(cf0_lpf[:minlen], axis=-1)
        mcep = mcep[:minlen]
        bap = bap[:minlen]

        # save features
        if 'feature_format' in config:
            extension = config['feature_format']
        else:
            extension = DEFAULT_HDF5_EXTENSION
        hdf5_out_name = generate_hdf5_out_path_from_input_name(wav_name, out_dir, extension)

        print(hdf5_out_name)
        write_hdf5(hdf5_out_name, "/uv", uv)
        write_hdf5(hdf5_out_name, "/f0", f0)
        write_hdf5(hdf5_out_name, "/cf0", cf0_lpf)
        write_hdf5(hdf5_out_name, "/mcep", mcep)
        write_hdf5(hdf5_out_name, "/bap", bap)

    queue.put("Finish")


def get_all_files_from_dir(dir_path: str) -> List[str]:
    all_dir_content = os.listdir(dir_path)
    all_files = []
    for elem in all_dir_content:
        absolute_path = os.path.join(dir_path, elem)
        if os.path.isfile(absolute_path):
            all_files.append(absolute_path)

    return all_files


def prepare_out_folder(out_folder_path: str):
    if os.path.exists(out_folder_path):
        # cleanup folder
        all_dir_content = os.listdir(out_folder_path)
        if len(all_dir_content) != 0:
            print("There are", len(all_dir_content), "objects in output dir. Cleaning it.")
        for elem in all_dir_content:
            absolute_path = os.path.join(out_folder_path, elem)
            if os.path.isfile(absolute_path):
                os.remove(absolute_path)
    else:
        os.makedirs(out_folder_path, exist_ok=True)


def get_filetitle_from_path(file_path: str) -> str:
    # Split the file path into directory path and filename
    directory, filename = os.path.split(file_path)

    # Split the filename into name and extension
    file_title, file_extension = os.path.splitext(filename)

    return file_title


def generate_hdf5_out_path_from_input_name(input_name: str, out_dir: str, extension: str) -> str:
    filetitle = get_filetitle_from_path(input_name)
    hdf5_out_path = os.path.join(out_dir, filetitle + extension)
    return hdf5_out_path


def read_yaml_config(path_to_config: str) -> dict:
    with open(path_to_config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print()
        except yaml.YAMLError as exc:
            print(exc)
    return config


def extract_features_main(input_dir: str, output_dir: str, path_to_config: str):
    config = read_yaml_config(path_to_config)

    # read list
    file_list = get_all_files_from_dir(input_dir)
    if len(file_list) == 0:
        raise Exception("Input directory " + input_dir + " is empty.")
    print("Number of inout files:", len(file_list))

    # list division
    file_lists = np.array_split(file_list, 10)
    file_lists = [f_list.tolist() for f_list in file_lists]
    configs = [config] * len(file_lists)

    # set mode
    target_fn = world_feature_extract

    # create folder
    prepare_out_folder(output_dir)

    # multi processing
    processes = []
    queue = mp.Queue()
    for f, _config in zip(file_lists, configs):
        p = mp.Process(
            target=target_fn,
            args=(queue, f, _config, output_dir),
        )
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Path where input files for feature extraction are located.")
    parser.add_argument("--output_dir", type=str, help="Path where extracted features will be located.")
    parser.add_argument("--path_to_config", type=str, default="configs/extract_features_default.yaml",
                        help="path to config with params related to feature extraction")
    args_ = parser.parse_args()

    extract_features_main(args_.input_dir, args_.output_dir, args_.path_to_config)
