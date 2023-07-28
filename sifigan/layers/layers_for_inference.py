from typing import List, Tuple

import torch
import torch.nn as nn

class FilterNetwork(nn.Module):
    def __init__(self, first_upsample: nn.Sequential, second_upsample: nn.Sequential, third_upsample: nn.Sequential,
                 fourth_upsample: nn.Sequential, first_process_block: nn.ModuleList,
                 second_process_block: nn.ModuleList, third_process_block: nn.ModuleList,
                 fourth_process_block: nn.ModuleList):
        super().__init__()
        self.first_upsample = first_upsample
        self.second_upsample = second_upsample
        self.third_upsample = third_upsample
        self.fourth_upsample = fourth_upsample

        self.num_proc_blocks = 3

        self.first_process_block = first_process_block
        self.second_process_block = second_process_block
        self.third_process_block = third_process_block
        self.fourth_process_block = fourth_process_block

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # first upsample
        c = self.first_upsample(input[0]) + input[-1]
        cs = torch.zeros(c.shape, dtype=c.dtype, device=c.device)  # initialize
        for j, fn_block in enumerate(self.first_process_block):
            cs += fn_block(c)
        c = cs / self.num_proc_blocks

        # second upsample
        c = self.second_upsample(c) + input[-2]
        cs = torch.zeros(c.shape, dtype=c.dtype, device=c.device)  # initialize
        for j, fn_block in enumerate(self.second_process_block):
            cs += fn_block(c)
        c = cs / self.num_proc_blocks

        # third upsample
        c = self.third_upsample(c) + input[-3]
        cs = torch.zeros(c.shape, dtype=c.dtype, device=c.device)  # initialize
        for j, fn_block in enumerate(self.third_process_block):
            cs += fn_block(c)
        c = cs / self.num_proc_blocks

        # fourth upsample
        c = self.fourth_upsample(c) + input[-4]
        cs = torch.zeros(c.shape, dtype=c.dtype, device=c.device)  # initialize
        for j, fn_block in enumerate(self.fourth_process_block):
            cs += fn_block(c)
        c = cs / self.num_proc_blocks

        return c
