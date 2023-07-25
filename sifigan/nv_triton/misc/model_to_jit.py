import argparse
import time

import torch

from sifigan.models import SiFiGANGenerator
from sifigan.nv_triton.misc.utils_funcs import remove_weight_norm, parse_bool


def convert_and_save_as_jit(checkpoint_path: str, save_path: str or None, fp16: bool = False):
    model = SiFiGANGenerator(in_channels=43, out_channels=1, channels=512, kernel_size=7,
                             upsample_scales=[5, 4, 3, 2], upsample_kernel_sizes=[10, 8, 6, 4])
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['model']['generator'])
    model.eval()
    remove_weight_norm(model)
    if fp16:
        model = model.half()
    print("Start tracing model")
    start_time = time.time()
    traced_model = torch.jit.script(model)
    print("Model successfully traced. Time taken:", str(time.time() - start_time))
    if save_path is not None:
        traced_model.save(save_path)
        print("Model successfully saved.")
    return traced_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chkp_path", type=str)
    parser.add_argument("save_path", type=str)
    parser.add_argument("--fp16", type=str, default='false')
    _args = parser.parse_args()

    convert_and_save_as_jit(_args.chkp_path, _args.save_path, parse_bool(_args.fp16))
