# -*- coding: utf-8 -*-

"""Decoding Script for Source-Filter HiFiGAN.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG

"""
import argparse
import os
import sys
from time import time
from typing import Tuple, List

import numpy as np
import soundfile as sf
import torch
import tritonclient.grpc as grpcclient
from hydra.utils import to_absolute_path
from tqdm import tqdm

from sifigan.datasets import FeatDataset
from sifigan.nv_triton.client.utils import read_yaml_config, prepare_out_folder
from sifigan.utils import SignalGenerator, dilated_factor


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_dataset(data_config: dict, f0_factor: float, input_dir: str) -> FeatDataset:
    dataset = FeatDataset(
        stats=data_config['stats'],
        feat_list=None,
        return_filename=True,
        sample_rate=data_config['sample_rate'],
        hop_size=data_config['hop_size'],
        aux_feats=data_config['aux_feats'],
        f0_factor=f0_factor,
        input_dir=input_dir
    )

    return dataset


def create_signal_generator(data_config: dict) -> SignalGenerator:
    signal_generator = SignalGenerator(
        sample_rate=data_config['sample_rate'],
        hop_size=data_config['hop_size'],
        sine_amp=data_config['sine_amp'],
        noise_amp=data_config['noise_amp'],
        signal_types=data_config['signal_types'],
    )

    return signal_generator


def create_triton_client() -> grpcclient.InferenceServerClient:
    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2
        )
        triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001',
            verbose=False,
            keepalive_options=keepalive_options)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    return triton_client


def prepare_input_data(in_signal: np.ndarray,
                       c: np.ndarray, dfs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.expand_dims(np.transpose(c, (1, 0)), axis=0)
    dfs = [np.expand_dims(np.expand_dims(df, axis=0), axis=0) for df in dfs]

    # do trick to support dynamic sized denses without using list in model
    true_lengths = np.zeros((in_signal.shape[0], len(dfs)), dtype=np.int64)

    max_length = - 1
    for df in dfs:
        if df.shape[-1] > max_length:
            max_length = df.shape[-1]

    for idx, df in enumerate(dfs):
        offset = max_length - df.shape[-1]
        true_lengths[:, idx] = df.shape[-1]
        if offset == 0:
            continue
        dfs[idx] = np.concatenate([df, np.zeros((*df.shape[:-1], offset), dtype=df.dtype)], axis=-1)

    # cat across 1-st axis
    dfs = np.concatenate(dfs, axis=1)

    return in_signal, c, dfs, true_lengths


def prepare_input_and_outputs_for_prediction(in_signal, c, dfs, true_lengths):
    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('INPUT__0', list(in_signal.shape), "FP32"))
    inputs.append(grpcclient.InferInput('INPUT__1', list(c.shape), "FP32"))
    inputs.append(grpcclient.InferInput('INPUT__2', list(dfs.shape), "FP32"))
    inputs.append(grpcclient.InferInput('INPUT__3', list(true_lengths.shape), "INT64"))

    # Initialize the data
    inputs[0].set_data_from_numpy(in_signal.astype(np.float32))
    inputs[1].set_data_from_numpy(c.astype(np.float32))
    inputs[2].set_data_from_numpy(dfs.astype(np.float32))
    inputs[3].set_data_from_numpy(true_lengths)

    outputs.append(grpcclient.InferRequestedOutput('OUTPUT__0'))

    return inputs, outputs


def decode_main(input_dir: str, output_dir: str, path_to_config: str) -> None:
    """Run decoding process."""

    config = read_yaml_config(path_to_config)

    # set up seed
    set_seed(int(config['seed']))

    # prepare_out_dir
    prepare_out_folder(output_dir, force_cleanup=True)

    # create triron client
    triton_client = create_triton_client()

    data_config = config['data']

    total_rtf = 0.0
    for f0_factor in config['f0_factors']:
        dataset = create_dataset(data_config, f0_factor, input_dir)
        print(f"The number of features to be decoded = {len(dataset)}.")

        signal_generator = create_signal_generator(data_config)

        with tqdm(dataset, desc="[decode]") as pbar:
            for idx, (feat_path, c, f0, cf0) in enumerate(pbar, 1):
                # create dense factors
                dfs = []
                for df, us in zip(
                    data_config['dense_factors'],
                    np.cumprod(config['generator']['upsample_scales']),
                ):
                    dfs += [
                        np.repeat(dilated_factor(cf0, data_config['sample_rate'], df), us)
                        if data_config['df_f0_type'] == "cf0"
                        else np.repeat(dilated_factor(f0, data_config['sample_rate'], df), us)
                    ]

                # create input signal
                if data_config['sine_f0_type'] == "cf0":
                    cf0 = torch.FloatTensor(cf0).view(1, 1, -1)
                    in_signal = signal_generator(cf0)
                    in_signal = in_signal.numpy()
                elif data_config['sine_f0_type'] == "f0":
                    f0 = torch.FloatTensor(f0).view(1, 1, -1)
                    in_signal = signal_generator(f0)
                    in_signal = in_signal.numpy()

                in_signal, c, dfs, true_lengths = prepare_input_data(in_signal, c, dfs)

                inputs, outputs = prepare_input_and_outputs_for_prediction(in_signal, c, dfs, true_lengths)

                # Test with outputs
                start = time()
                results = triton_client.infer(model_name='sifigan',
                                              inputs=inputs,
                                              outputs=outputs,
                                              headers={'test': '1'})
                print("Prediction taken:", time() - start, "seconds")

                # # perform decoding
                # start = time()
                # outs = model(in_signal, c, dfs, true_lengths)
                # print('in_signal shape', in_signal.shape)
                # print('c shape', c.shape)
                # print('dfs shape', dfs.shape)
                # print('true_lengths shape', true_lengths.shape)
                y = results
                # print("!!!!!", torch.isnan(y).any())
                rtf = (time() - start) / (y.size(-1) / config.data.sample_rate)
                pbar.set_postfix({"RTF": rtf})
                total_rtf += rtf

                # save output signal as PCM 16 bit wav file
                utt_id = os.path.splitext(os.path.basename(feat_path))[0]
                save_path = os.path.join(output_dir, f"{utt_id}_f{f0_factor:.2f}.wav")
                y = y.view(-1).cpu().numpy()
                sf.write(save_path, y, config.data.sample_rate, "PCM_16")

            # report average RTF
            print(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.4f}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Path where input files for feature extraction are located.")
    parser.add_argument("--output_dir", type=str, help="Path where extracted features will be located.")
    parser.add_argument("--path_to_config", type=str, default="configs/decode_default.yaml",
                        help="path to config with params related to feature extraction")
    args_ = parser.parse_args()

    decode_main(args_.input_dir, args_.output_dir, args_.path_to_config)
