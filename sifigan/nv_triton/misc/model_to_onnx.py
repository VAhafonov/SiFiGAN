import argparse
from typing import Tuple

import torch

from sifigan.models import SiFiGANGenerator
from sifigan.nv_triton.misc.model_to_jit import remove_weight_norm


def read_and_preprocess_input_tensors(test_tensor_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dict_ = torch.load(test_tensor_path, map_location='cpu')

    in_signal = dict_['in_signal'].cuda()
    c = dict_['c'].cuda()
    dfs = dict_['dfs'].cuda()
    true_length = dict_['true_lengths'].cuda()

    return in_signal, c, dfs, true_length


def convert_and_save_as_onnx(checkpoint_path: str, save_path: str, test_tensor_path: str):
    # traced_model = convert_and_save_as_jit(checkpoint_path, save_path=None)
    model = SiFiGANGenerator(in_channels=43, out_channels=1, channels=512, kernel_size=7,
                             upsample_scales=[5, 4, 3, 2], upsample_kernel_sizes=[10, 8, 6, 4])
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['model']['generator'])
    model.eval()
    model.cuda()
    remove_weight_norm(model)
    in_signal, c, dfs, true_length = read_and_preprocess_input_tensors(test_tensor_path)

    torch.onnx.export(model,
                      args=(in_signal, c, dfs, true_length),
                      f=save_path,
                      input_names=["INPUT__0", "INPUT__1", "INPUT__2", "INPUT__3"],
                      output_names=["OUTPUT__0"],
                      dynamic_axes={
                          "INPUT__0": {0: "batch_size", 1: "input_dim_0_1", 2: "input_dim_0_2"},
                          "INPUT__1": {0: "batch_size", 1: "input_dim_1_1", 2: "input_dim_1_2"},
                          "INPUT__2": {0: "batch_size", 1: "input_dim_2_1", 2: "input_dim_2_2"},
                          "INPUT__3": {0: "batch_size", 1: "input_dim_3_1"},
                          "OUTPUT__0": {0: "batch_size", 1: "output_dim_0_1"},
                      },
                      verbose=True,
                      export_params=True
                      )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chkp_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("test_tensor_path", type=str)
    _args = parser.parse_args()

    convert_and_save_as_onnx(_args.chkp_path, _args.save_path, _args.test_tensor_path)