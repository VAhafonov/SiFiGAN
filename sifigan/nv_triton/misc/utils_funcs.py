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
