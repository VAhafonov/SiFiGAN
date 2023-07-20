# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Residual block modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG
    - https://github.com/r9y9/wavenet_vocoder

"""

from logging import getLogger

import torch
import torch.nn as nn
from sifigan.layers import Snake
from sifigan.utils import index_initial, pd_indexing, index_initial_for_jit

# A logger for this file
logger = getLogger(__name__)

# add ModuleInterface
@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor: # `input` has a same name in Sequential forward
        pass


class Conv1d(nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias=True):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class Conv2d(nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Conv2d1x1(Conv2d):
    """1x1 Conv2d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias=True):
        """Initialize 1x1 Conv2d module."""
        super(Conv2d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class ResidualBlock(nn.Module):
    """Residual block module in HiFiGAN."""

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 3, 5),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            dilations (List[int]): List of dilation factors.
            use_additional_convs (bool): Whether to use additional convolution layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.

        """
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = nn.ModuleList()
        if use_additional_convs:
            self.convs2 = nn.ModuleList()
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        for dilation in dilations:
            if nonlinear_activation == "Snake":
                nonlinear = Snake(channels, **nonlinear_activation_params)
            else:
                nonlinear = getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
            self.convs1 += [
                nn.Sequential(
                    nonlinear,
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        bias=bias,
                        padding=(kernel_size - 1) // 2 * dilation,
                    ),
                )
            ]
            if use_additional_convs:
                if nonlinear_activation == "Snake":
                    nonlinear = Snake(channels, **nonlinear_activation_params)
                else:
                    nonlinear = getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
                self.convs2 += [
                    nn.Sequential(
                        nonlinear,
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, input: torch.Tensor):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        for idx in range(len(self.convs1)):
            submodule: ModuleInterface = self.convs1[idx]
            xt = submodule.forward(input)
            # if self.use_additional_convs:
            #     submodule1: ModuleInterface = self.convs2[idx]
            #     xt = submodule1.forward(xt)
            input = xt + input
        return input


class AdaptiveResidualBlock(nn.Module):
    """Residual block module in HiFiGAN."""

    def __init__(
        self,
        kernel_size=3,
        channels=512,
        dilations=(1, 2, 4),
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.

        """
        super().__init__()
        self.use_additional_convs = use_additional_convs
        assert kernel_size == 3, "Currently only kernel_size = 3 is supported."
        self.channels = channels
        self.dilations = dilations
        self.nonlinears = nn.ModuleList()
        self.convsC = nn.ModuleList()
        self.convsP = nn.ModuleList()
        self.convsF = nn.ModuleList()
        if use_additional_convs:
            self.convsA = nn.ModuleList()
        for _ in dilations:
            if nonlinear_activation == "Snake":
                self.nonlinears += [Snake(channels, **nonlinear_activation_params)]
            else:
                self.nonlinears += [getattr(nn, nonlinear_activation)(**nonlinear_activation_params)]
            self.convsC += [
                Conv1d1x1(
                    channels,
                    channels,
                    bias=bias,
                ),
            ]
            self.convsP += [
                Conv1d1x1(
                    channels,
                    channels,
                    bias=bias,
                ),
            ]
            self.convsF += [
                Conv1d1x1(
                    channels,
                    channels,
                    bias=bias,
                ),
            ]
            if use_additional_convs:
                if nonlinear_activation == "Snake":
                    nonlinear = Snake(channels, **nonlinear_activation_params)
                else:
                    nonlinear = getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
                self.convsA += [
                    nn.Sequential(
                        nonlinear,
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size,
                            dilation=1,
                            bias=bias,
                            padding=(kernel_size - 1) // 2,
                        ),
                    )
                ]

    def forward(self, x, d):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            d (Tensor): Input pitch-dependent dilated factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        for i, dilation in enumerate(self.dilations):
            assert isinstance(i, int)
            submodule: ModuleInterface = self.nonlinears[i]
            xt = submodule.forward(x)

            xP, xF = pd_indexing(xt, d, dilation)

            submodule1: ModuleInterface = self.convsC[i]
            submodule2: ModuleInterface = self.convsP[i]
            submodule3: ModuleInterface = self.convsF[i]
            xt = submodule1.forward(xt) + submodule2.forward(xP) + submodule3.forward(xF)

            if self.use_additional_convs:
                submodule4: ModuleInterface = self.convsA[i]
                xt = submodule4.forward(xt)
            x = xt + x
        return x
