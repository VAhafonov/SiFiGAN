from typing import NamedTuple, Tuple

import torch.nn
from torch.nn.utils.weight_norm import WeightNorm


# workaround from https://github.com/pytorch/pytorch/issues/57289
def remove_weight_norm(module: torch.nn.Module):
    module_list = [mod for mod in module.children()]
    if len(module_list) == 0:
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook.remove(module)
                del module._forward_pre_hooks[k]
    else:
        for mod in module_list:
            remove_weight_norm(mod)


class InputData(NamedTuple):
    in_signal: torch.Tensor
    c: torch.Tensor
    dfs: torch.Tensor


class OutputData(NamedTuple):
    y: torch.Tensor


def read_and_preprocess_test_tensors(test_tensor_path: str, do_read_output_tensor: bool = False,
                                     do_convert_to_cuda: bool = False,
                                     fp16: bool = False) -> InputData or Tuple[InputData, OutputData]:
    dict_ = torch.load(test_tensor_path, map_location='cpu')

    in_signal = dict_['in_signal']
    c = dict_['c']
    dfs = dict_['dfs']

    if fp16:
        in_signal = in_signal.half()
        c = c.half()
        dfs = dfs.half()

    if do_convert_to_cuda:
        in_signal = in_signal.cuda()
        c = c.cuda()
        dfs = dfs.cuda()

    input_data = InputData(in_signal, c, dfs)

    if do_read_output_tensor:
        y = dict_['y']
        if do_convert_to_cuda:
            y = y.cuda()
        return input_data, OutputData(y)

    return input_data


def parse_bool(val: str) -> bool:
    lower = val.lower()
    if lower == 'true':
        return True
    elif lower == 'false':
        return False
    raise ValueError('Invalid boolean: ' + val)
