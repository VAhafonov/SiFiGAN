import argparse
import time

import torch

from sifigan.models import SiFiGANGenerator
from sifigan.nv_triton.misc.utils_funcs import remove_weight_norm, read_and_preprocess_test_tensors, parse_bool


def convert_and_save_as_onnx(checkpoint_path: str, save_path: str, test_tensor_path: str,
                             use_dynamic_shape: bool = True, fp16: bool = False):
    model = SiFiGANGenerator(in_channels=43, out_channels=1, channels=512, kernel_size=7,
                             upsample_scales=[5, 4, 3, 2], upsample_kernel_sizes=[10, 8, 6, 4])
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['model']['generator'])
    model.eval()
    model.cuda()
    remove_weight_norm(model)
    model.apply_layer_tweaks()
    if fp16:
        model = model.half()
    input_data = read_and_preprocess_test_tensors(test_tensor_path, do_read_output_tensor=False,
                                                  do_convert_to_cuda=True, fp16=fp16)
    in_signal, c, dfs = input_data

    print("Start onnx export")
    start_time = time.time()
    if use_dynamic_shape:
        torch.onnx.export(model,
                          args=(in_signal, c, dfs),
                          f=save_path,
                          input_names=["INPUT__0", "INPUT__1", "INPUT__2"],
                          output_names=["OUTPUT__0"],
                          dynamic_axes={
                              "INPUT__0": {0: "batch_size", 2: "input_dim_0_2"},
                              "INPUT__1": {0: "batch_size",  2: "input_dim_1_2"},
                              "INPUT__2": {0: "batch_size",  2: "input_dim_2_2"},
                              "OUTPUT__0": {0: "batch_size", 2: "output_dim_0_2"},
                          },
                          verbose=False,
                          export_params=True
                          )
    else:
        torch.onnx.export(model,
                          args=(in_signal, c, dfs),
                          f=save_path,
                          input_names=["INPUT__0", "INPUT__1", "INPUT__2"],
                          output_names=["OUTPUT__0"],
                          verbose=False,
                          export_params=True
                          )
    print("Done onnx export. Time taken:", str(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chkp_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("test_tensor_path", type=str)
    parser.add_argument("--use_dynamic_shape", type=str, default='true')
    parser.add_argument("--fp16", type=str, default='false')
    _args = parser.parse_args()

    convert_and_save_as_onnx(_args.chkp_path, _args.save_path,
                             _args.test_tensor_path, parse_bool(_args.use_dynamic_shape), parse_bool(_args.fp16))
