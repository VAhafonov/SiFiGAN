# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Indexing-related functions."""
from typing import List

import torch


def pd_indexing(x: torch.Tensor, d: torch.Tensor, dilation: int):
    """Pitch-dependent indexing of past and future samples.

    Args:
        x (Tensor): Input feature map (B, C, T).
        d (Tensor): Input pitch-dependent dilated factors (B, 1, T).
        dilation (Int): Dilation size.
        batch_index (Tensor): Batch index
        ch_index (Tensor): Channel index

    Returns:
        Tensor: Past output tensor (B, out_channels, T)
        Tensor: Future output tensor (B, out_channels, T)

    """
    B, C, T = x.size()
    batch_index = torch.arange(0, B, dtype=torch.long, device=x.device).reshape(B, 1, 1)
    ch_index = torch.arange(0, C, dtype=torch.long, device=x.device).reshape(1, C, 1)
    dilations = torch.clamp((d * dilation).long(), min=1)

    # get past index (assume reflect padding)
    idx_base = torch.arange(0, T, dtype=torch.long, device=x.device).reshape(1, 1, T)
    idxP = (idx_base - dilations).abs() % T
    idxP = (batch_index, ch_index, idxP)
    idxP_idx: List[List[List[List[int]]]] = []
    for elem in idxP:
        inner_list: List[List[List[int]]] = elem.tolist()
        idxP_idx.append(inner_list)

    # temp
    indexed_x_p = x[idxP_idx]

    # get future index (assume reflect padding)
    idxF = idx_base + dilations
    overflowed = idxF >= T
    idxF[overflowed] = -(idxF[overflowed] % T)
    idxF = (batch_index, ch_index, idxF)
    idxF_idx: List[List[List[List[int]]]] = []
    for elem in idxF:
        inner_list: List[List[List[int]]] = elem.tolist()
        idxF_idx.append(inner_list)

    # return x[list(idxP_idx)], x[list(idxF_idx)]
    return indexed_x_p, x[idxF_idx]


def index_initial(n_batch, n_ch, tensor=True):
    """Tensor batch and channel index initialization.

    Args:
        n_batch (Int): Number of batch.
        n_ch (Int): Number of channel.
        tensor (bool): Return tensor or numpy array

    Returns:
        Tensor: Batch index
        Tensor: Channel index

    """
    batch_index = []
    for i in range(n_batch):
        batch_index.append([[i]] * n_ch)
    ch_index = []
    for i in range(n_ch):
        ch_index += [[i]]
    ch_index = [ch_index] * n_batch

    if tensor:
        batch_index = torch.tensor(batch_index)
        ch_index = torch.tensor(ch_index)
        if torch.cuda.is_available():
            batch_index = batch_index.cuda()
            ch_index = ch_index.cuda()
    return batch_index, ch_index


def index_initial_for_jit(n_batch: int, n_ch: int):
    """Tensor batch and channel index initialization.

    Args:
        n_batch (Int): Number of batch.
        n_ch (Int): Number of channel.
        tensor (bool): Return tensor or numpy array

    Returns:
        Tensor: Batch index
        Tensor: Channel index

    """
    assert isinstance(n_ch, int)
    assert isinstance(n_batch, int)
    batch_index = []
    for i in torch.arange(n_batch):
        batch_index.append(torch.full([1, n_ch, 1], fill_value=i, dtype=torch.int64, device='cpu'))
    batch_index = torch.cat(batch_index, dim=0)

    ch_index = torch.unsqueeze(torch.unsqueeze(torch.arange(0, n_ch, dtype=torch.int64, device='cpu'), dim=0), dim=-1)
    if n_batch != 1:
        ch_index = torch.cat([ch_index * n_batch], dim=0)

    return batch_index, ch_index
