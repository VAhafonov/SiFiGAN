import argparse

import numpy as np
import torch.jit

from sifigan.nv_triton.misc.utils_funcs import read_and_preprocess_test_tensors, parse_bool


def evaluate_jit_main(jit_model_path: str, test_tensor_path: str, fp16: bool = False):
    # load model
    jit_model = torch.jit.load(jit_model_path)
    # load test data
    input_data, output_data = read_and_preprocess_test_tensors(test_tensor_path, do_read_output_tensor=True,
                                                               do_convert_to_cuda=True, fp16=fp16)
    # predict
    jit_output = jit_model(input_data.in_signal, input_data.c, input_data.dfs)

    # compare tensors
    jit_output_np = jit_output.detach().numpy()
    target_output = output_data.y.numpy()
    if fp16:
        # convert jit output back to float32
        jit_output_np = jit_output_np.astype(np.float32)
    are_tensors_equal = np.allclose(jit_output_np, target_output, equal_nan=True, atol=1e-5)
    print("Tensors are equal:", are_tensors_equal)
    print("Max diff in tensors:",
          np.max(np.abs(jit_output_np[~np.isnan(jit_output_np)] - target_output[~np.isnan(target_output)])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("jit_model_path", type=str)
    parser.add_argument("test_tensor_path", type=str)
    parser.add_argument("--fp16", type=str, default='false')
    _args = parser.parse_args()

    evaluate_jit_main(_args.jit_model_path, _args.test_tensor_path, parse_bool(_args.fp16))
