import argparse

import numpy as np
import onnxruntime

from sifigan.nv_triton.misc.utils_funcs import read_and_preprocess_test_tensors


def evaluate_onnx_main(onnx_model_path: str, test_tensor_path: str):
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    input_data, output_data = read_and_preprocess_test_tensors(test_tensor_path, do_read_output_tensor=True,
                                                               do_convert_to_cuda=False)
    # Run the ONNX model with the input data
    outputs = ort_session.run(None, {"INPUT__0": input_data.in_signal.numpy(),
                                     "INPUT__1": input_data.c.numpy(),
                                     "INPUT__2": input_data.dfs.numpy()})
    # compare tensors
    onnx_output_np = outputs[0]
    target_output = output_data.y.numpy()
    are_tensors_equal = np.allclose(onnx_output_np, target_output, equal_nan=True, atol=1e-5)
    print("Tensors are equal:", are_tensors_equal)
    print("Max diff in tensors:",
          np.max(np.abs(onnx_output_np[~np.isnan(onnx_output_np)] - target_output[~np.isnan(target_output)])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model_path", type=str)
    parser.add_argument("test_tensor_path", type=str)
    _args = parser.parse_args()

    evaluate_onnx_main(_args.onnx_model_path, _args.test_tensor_path)
