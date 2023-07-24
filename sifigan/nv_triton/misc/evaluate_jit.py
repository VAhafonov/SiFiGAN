import argparse

import torch.jit

from sifigan.nv_triton.misc.utils_funcs import read_and_preprocess_test_tensors


def evaluate_jit_main(jit_model_path: str, test_tensor_path: str):
    jit_model = torch.jit.load(jit_model_path)
    input_data, output_data = read_and_preprocess_test_tensors(test_tensor_path, do_read_output_tensor=True,
                                                               do_convert_to_cuda=True)
    jit_output = jit_model(input_data.in_signal, input_data.c, input_data.dfs, input_data.true_length)
    print(jit_output)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("jit_model_path", type=str)
    parser.add_argument("test_tensor_path", type=str)
    _args = parser.parse_args()

    evaluate_jit_main(_args.chkp_path, _args.save_path)
