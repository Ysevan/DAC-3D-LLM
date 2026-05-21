# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

from einops import rearrange
from typing import Optional

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
    "YOLO_FPRS_GA",
    "CoDA",
    "MSGI_FCM",
    "FPRS_GA",
    "DW_GCA",
    "FG_RCA",
    "MiLKConvAttn",
    "FreqAttnPlus",
    "InterRowColSelfAttention",
    "EnhancedEMA",
    "SelfAttentionSEBlock",
    "FullyDynamicSEBlock",
    "DACModule",
    "SPRSA",
    "CGHalfConv_GS",
    "CoordAttMeanMax",
    "MultiScaleFreqDenoise",
    "GTPStem",
    "PriorBypass",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        # self.m = nn.ModuleList(
        #     DW_GCA_Block(self.c, self.c, shortcut, g) for _ in range(n)
        # )
        # self.m = nn.ModuleList(
        #     FPRS_GA_Block(self.c, self.c, shortcut, g) for _ in range(n)
        # )

        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)




class FPRS_GA(nn.Module):
    """
    FPRS-GA（Frequency-guided Pixel Refinement with Spectral Gated Aggregation）
    设计动机（人话版）：
    1）空间域：深度可分卷积 + 1x1 卷积，低成本抓局部纹理，这一支路等价于SPR-SA的“像素精炼”主干。
    2）频域：FFT把特征映射到频谱，把实部/虚部当成2个通道堆起来，用1x1卷积“调制频谱”，再IFFT回空间。
    3）门控聚合：引入可学习系数 gamma/beta，频域与空间域动态平衡，避免频域过强带来的伪影回灌。
    4）通道注意力：GAP + Softmax 做通道权重，对应论文里“每个像素在同一位置感知不同退化信号”的直觉。
    """
    def __init__(self, dim: int, growth_rate: float = 2.0):
        super().__init__()
        # 隐藏通道（和原SPR-SA保持一致策略）
        hidden_dim = int(dim * growth_rate)

        # ===== 空间域：局部像素精炼（CV缝合救星保留件） =====
        # 深度可分卷积提局部、1x1卷积做通道混合
        self.local_refine = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=dim),  # DWConv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0)       # PWConv
        )

        # ===== 频域：实/虚拼接 → 频谱调制 → 复原复数谱 → IFFT =====
        # 注意：在频域我们不做复数卷积，而是把 real/imag 当作 2*hidden_dim 个实通道来做线性变换
        self.freq_proj = nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=1, stride=1, padding=0)

        # ===== 通道注意力（GAP + Softmax），就是CV缝合救星的“通道加权法宝” =====
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0)

        # 频域/空间门控系数（可学习），初始化小一点更稳（频域先浅尝辄止）
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 频域分支权重
        self.beta  = nn.Parameter(torch.tensor(1.0))  # 空间分支权重

    @torch.no_grad()
    def _debug_shapes(self, name, x):
        # 可选：调试时打开，打印形状
        # print(f"[debug] {name}: {tuple(x.shape)}")
        pass

    def _fft_modulate(self, x_local: torch.Tensor) -> torch.Tensor:
        """
        频域路径：
        1) FFT 得到复数谱 X（B,C,H,W），分解为 real/imag 两个实张量
        2) 在 [real, imag] 拼接的 2C 通道上做 1x1 卷积进行频谱“调制”
        3) 把调制后的实/虚再组成复数谱 X'，IFFT 回空间，取 real 作为增强特征
        """
        # X: 复数谱（complex64/complex32）
        X = torch.fft.fft2(x_local, norm='ortho')
        real = X.real
        imag = X.imag
        self._debug_shapes("fft_real", real)
        self._debug_shapes("fft_imag", imag)

        # 通道拼接 [B, 2C, H, W]
        freq_cat = torch.cat([real, imag], dim=1)

        # 1x1 卷积做频谱调制（相当于对实/虚共同做线性组合）
        freq_mod = self.freq_proj(freq_cat)

        # 切回实/虚
        c = freq_mod.shape[1] // 2
        real_mod, imag_mod = freq_mod[:, :c], freq_mod[:, c:]

        # 复原复数谱，IFFT回空间
        X_mod = torch.complex(real_mod, imag_mod)
        x_freq = torch.fft.ifft2(X_mod, norm='ortho').real  # 只取实部，数值更稳
        return x_freq

    def _channel_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        通道注意力（GAP + Softmax），让每个通道学会“自我权重”
        这一步就是经典的“CV缝合救星”通道重加权——既简单、又好使。
        """
        w = F.adaptive_avg_pool2d(x, (1, 1))           # [B,C,1,1]
        w = F.softmax(w, dim=1)                        # 通道维归一化
        return x * w                                   # 按通道乘权

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：x [B, dim, H, W]
        输出：y [B, dim, H, W]
        """
        # 1) 空间局部精炼（低成本、强鲁棒）：DWConv + PWConv
        x_local = self.local_refine(x)                 # [B, hidden, H, W]
        self._debug_shapes("x_local", x_local)

        # 2) 频域增强：FFT → 频谱调制 → IFFT（只取real）
        x_freq = self._fft_modulate(x_local)           # [B, hidden, H, W]
        self._debug_shapes("x_freq", x_freq)

        # 3) 通道注意力（CV缝合救星的拿手戏）
        x_local = self._channel_attention(x_local)     # [B, hidden, H, W]

        # 4) 频-空门控聚合（Spectral Gated Aggregation）
        fused = self.gamma * x_freq + self.beta * x_local

        # 5) 激活 + 映射回输入通道数
        fused = self.act(fused)
        y = self.out_proj(fused)                       # [B, dim, H, W]
        return y


class YOLO_FPRS_GA(FPRS_GA):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # 1. 调用父类 FPRS_GA 的 __init__ 方法 (它会调用 nn.Module.__init__)
        # 这一步必须是第一个执行的！
        # FPRS_GA 核心模块 dim 必须是 c2
        super().__init__(dim=c2, growth_rate=2.0)

        # 2. 现在可以安全地创建并赋值 PyTorch Module 属性了
        # 通道转换层：负责将 c1 通道的输入压缩或扩展到 c2 通道
        self.compress_expand_conv = Conv(c1, c2, k=1)

        # 保存参数（非 Module 属性，顺序不严格，但通常放最后）
        self.c1 = c1
        self.c2 = c2

    def forward(self, x):
        # ... (forward 逻辑保持不变)
        if self.c1 != self.c2:
            x = self.compress_expand_conv(x)
        return super().forward(x)

class FPRS_GA_Block(nn.Module):
    """Wrapper to make FPRS_GA behave like a YOLO Bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        assert c1 == c2, "FPRS_GA only supports c1 == c2"

        self.fprs = FPRS_GA(c1)

        self.add = shortcut

    def forward(self, x):
        y = self.fprs(x)
        return x + y if self.add else y

# =========================================
# DW-GCA: Directional Wavelet Gated Cross-band Attention
# 方向性小波门控跨子带注意力（CVPR风格命名）
# 在原始 WaveletAttention 基础上加入：
# 1) 子带方向性门控（为 LH/HL/HH 单独学习门）
# 2) 跨子带注意力（在每个空间位置上，学习3个子带的加权分配）
# 3) 可学习子带再分配（将融合特征按通道再映射回 LH/HL/HH 以便 IDWT）
# 4) 混合通道注意力（SE + Softmax 归一化）与可学习残差融合
# =========================================

# =========================
# 构造 Daubechies-1 (Haar) 小波核
# =========================
def build_wavelet_kernels(device=None, dtype=torch.float32):
    """
    返回 2x2 的四个 2D 分析核：LL, LH, HL, HH
    对于 Db1 (Haar)：h0=[1/sqrt2, 1/sqrt2], h1=[-1/sqrt2, 1/sqrt2]
    2D 核是外积：h_row^T * h_col
    """
    s = 1.0 / math.sqrt(2.0)
    h0 = torch.tensor([s, s], dtype=dtype, device=device)      # 低通
    h1 = torch.tensor([-s, s], dtype=dtype, device=device)     # 高频
    # 外积得到 2x2 核
    LL = torch.ger(h0, h0)  # 低-低
    LH = torch.ger(h0, h1)  # 低-高（垂直边更敏感）
    HL = torch.ger(h1, h0)  # 高-低（水平边更敏感）
    HH = torch.ger(h1, h1)  # 高-高（对角边/角点）
    # 形状统一为 (4,1,2,2) 方便后续扩展到 groups=C
    filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)
    return filt  # (4,1,2,2)


class SE(nn.Module):
    """
    轻量通道注意力（Squeeze-Excitation风格）
    - 使用 GAP 降维 -> 两层1x1线性 -> Sigmoid
    - 用于与 Softmax 通道归一化互补：SE 强调“绝对重要性”，Softmax 强调“相对分配”
    """
    def __init__(self, channels, r=8):
        super().__init__()
        hidden = max(channels // r, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: [B,C,H,W]
        z = F.adaptive_avg_pool2d(x, 1)         # [B,C,1,1]
        z = F.relu(self.fc1(z), inplace=True)
        z = torch.sigmoid(self.fc2(z))          # [B,C,1,1]
        return z


class DW_GCA(nn.Module):
    """
    DW-GCA：方向性小波门控跨子带注意力
    总流程：
    1) DWT: X -> (LH, HL, HH, LL)
    2) 高频软阈值（可学习阈值，按通道缩放到每个子带的能量尺度）
    3) 方向性门控：为 LH/HL/HH 学习到通道级别的 gate（Sigmoid）
    4) 跨子带注意力（Cross-band Attention）：
       - 将 LH/HL/HH reshape 为 [B,C,3,h,w]，对 dim=2 做 softmax，得到逐像素的子带权重
       - 对三个子带进行加权和，得到融合高频 H_mix（强调当前像素更重要的方向）
    5) 可学习子带再分配：
       - 对每个通道学习 w_LH, w_HL, w_HH（softmax到3个），将 H_mix 再分配回 LH2/HL2/HH2 以便 IDWT
    6) IDWT：用 (LH2, HL2, HH2, LL) 重建 X_re
    7) 通道注意力（SE） + Softmax 通道归一化 的混合权重
    8) 残差融合： out = x * w_softmax + gamma * X_re * w_se
       gamma 为可学习标量，控制重建分支对最终输出的贡献
    """
    def __init__(self, channels, use_fc=True, se_ratio=8):
        super().__init__()
        self.channels = channels
        self.use_fc = use_fc

        # ---------- 高频软阈值参数（3个子带 * C） ----------
        # 用 Sigmoid 将其约束到 [0,1]，再自适应缩放到子带的均值幅度
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))

        # ---------- 方向性门控（3个子带 * C），Sigmoid ----------
        # 用于在阈值后对不同方向的响应进行通道级再加权
        self.dir_gate = nn.Parameter(torch.zeros(3, channels, 1, 1))

        # ---------- 可选的 FC（对 GAP 后的通道向量做线性变换，便于Softmax通道归一化前的可学习投影） ----------
        if use_fc:
            self.fc = nn.Linear(channels, channels, bias=True)

        # ---------- 可学习的子带再分配系数（C x 3），用于将融合后的 H_mix 映射回 LH/HL/HH ----------
        # 用 softmax 保证三者之和为1，提高可解释性与稳定性
        self.sub_redistribute = nn.Parameter(torch.zeros(channels, 3))  # 初始化为0，softmax后约等于均分

        # ---------- 通道注意力（SE） ----------
        self.se = SE(channels, r=se_ratio)

        # ---------- 可学习残差系数 ----------
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # ---------- 小波核（注册为 buffer，不参与训练） ----------
        filt = build_wavelet_kernels()
        self.register_buffer("w_analysis", filt)   # (4,1,2,2)
        self.register_buffer("w_synthesis", filt)  # Haar 正交：合成=分析

    # ---------- DWT 与 IDWT ----------
    def dwt(self, x):
        """
        x: (B,C,H,W)
        返回：LH, HL, HH, LL
        """
        B, C, H, W = x.shape

        # 填充到偶数尺寸，保证stride=2整除（避免边界丢失）
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        # 组卷积：每个通道使用同一组 4 个滤波器
        weight = self.w_analysis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)  # (B,4C,H/2,W/2)

        # 按子带拆分： [LL, LH, HL, HH] 顺序与上面 build 函数保持一致
        y = y.view(B, C, 4, y.size(-2), y.size(-1)).contiguous()
        LL = y[:, :, 0]  # (B,C,h,w)
        LH = y[:, :, 1]
        HL = y[:, :, 2]
        HH = y[:, :, 3]
        return LH, HL, HH, LL

    def idwt(self, LH, HL, HH, LL):
        """
        逆变换：将四个子带重建为 (B,C,H,W)
        """
        B, C, h, w = LL.shape
        # 将 4 个子带 stack 回 (B,4C,h,w)
        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4 * C, h, w)

        # 反卷积作为合成滤波器，stride=2
        weight = self.w_synthesis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        x_rec = F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)
        return x_rec

    # ---------- 高频软阈值 ----------
    @staticmethod
    def soft_threshold(x, thr):
        # 经典 soft-shrinkage： sign(x) * relu(|x| - thr)
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    def forward(self, x):
        """
        x: [B,C,H,W]
        输出：与输入同形状 [B,C,H,W]
        """
        B, C, H, W = x.shape

        # 1) 小波分解
        LH, HL, HH, LL = self.dwt(x)

        # 2) 高频子带软阈值（自适应按通道尺度化阈值）
        eps = 1e-6
        m_LH = LH.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HL = HL.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HH = HH.abs().mean(dim=(2, 3), keepdim=True) + eps

        t = torch.sigmoid(self.theta)  # (3,C,1,1), 映射到 0~1
        thr_LH = t[0].unsqueeze(0) * m_LH
        thr_HL = t[1].unsqueeze(0) * m_HL
        thr_HH = t[2].unsqueeze(0) * m_HH

        LH_hat = self.soft_threshold(LH, thr_LH)
        HL_hat = self.soft_threshold(HL, thr_HL)
        HH_hat = self.soft_threshold(HH, thr_HH)

        # 3) 方向性门控（通道级）：为 LH/HL/HH 引入 Sigmoid 门
        g = torch.sigmoid(self.dir_gate)  # (3,C,1,1)
        LH_hat = LH_hat * g[0].unsqueeze(0)   # 加强/抑制对“垂直边”更敏感的通道
        HL_hat = HL_hat * g[1].unsqueeze(0)   # 加强/抑制对“水平边”更敏感的通道
        HH_hat = HH_hat * g[2].unsqueeze(0)   # 加强/抑制对“对角/角点”更敏感的通道

        # 4) 跨子带注意力（逐像素地决定更信任哪个方向）
        # 将三个子带堆成 [B, C, 3, h, w]，对 dim=2 做 softmax，得到跨子带权重
        h = LH_hat.size(-2)
        w = LH_hat.size(-1)
        stack3 = torch.stack([LH_hat, HL_hat, HH_hat], dim=2)  # [B,C,3,h,w]
        # a: [B,C,3,h,w] -> softmax over band-dimension
        attn_band = F.softmax(stack3.abs().mean(dim=1, keepdim=True), dim=2)  # 用跨通道的能量引导（更稳定）
        # 也可以直接对 stack3 做一个 1x1x1 的线性映射后 softmax，这里用能量引导更轻量

        # 按权重加权求和得到融合高频 H_mix（仍为 [B,C,h,w]）
        H_mix = (stack3 * attn_band).sum(dim=2)  # [B,C,h,w]

        # 5) 可学习子带再分配：将 H_mix 映射回 LH2/HL2/HH2，保证可以做 IDWT
        # 对每个通道有3个权重，softmax 到 3 个子带
        w_redis = F.softmax(self.sub_redistribute, dim=1)  # [C,3]
        # reshape 为 [1,C,3,1,1] 便于广播
        w_redis = w_redis.view(1, C, 3, 1, 1)
        # 将 H_mix 拓展成3份，然后乘以通道的再分配系数
        H_mix_exp = H_mix.unsqueeze(2)  # [B,C,1,h,w]
        H_redist = H_mix_exp * w_redis  # [B,C,3,h,w]
        # 拆回三路
        LH2 = H_redist[:, :, 0]
        HL2 = H_redist[:, :, 1]
        HH2 = H_redist[:, :, 2]

        # 6) 逆小波重建，得到重建特征 X_re
        X_re = self.idwt(LH2, HL2, HH2, LL)  # [B,C,H',W'] 尺寸可能大于 H, W

        # --- 修复步骤：将重建特征裁剪回原始输入尺寸 ---
        # 确保 X_re 与 x 的 H/W 尺寸完全匹配
        _, _, H, W = x.shape
        # 使用 F.interpolate 或直接切片
        X_re = X_re[..., :H, :W]

        # 7) 通道注意力：SE（绝对重要性） + Softmax（相对分配）
        # 注意：这里的 X_re 已经裁剪
        # 7.1 Softmax 通道归一化（可选FC增强可分性）
        gap_vec = F.adaptive_avg_pool2d(X_re, 1).view(B, C)  # [B,C]
        if self.use_fc:
            gap_vec = self.fc(gap_vec)  # [B,C]
        w_softmax = F.softmax(gap_vec, dim=1).view(B, C, 1, 1)  # [B,C,1,1]

        # 7.2 SE 通道权重（Sigmoid）
        w_se = self.se(X_re)  # [B,C,1,1]

        # 8) 残差融合（现在 x 和 X_re 尺寸相同，融合不会报错）
        out = x * w_softmax + self.gamma * (X_re * w_se)
        return out

class DW_GCA_Block(nn.Module):
    """Wrapper to make FPRS_GA behave like a YOLO Bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        assert c1 == c2, "FPRS_GA only supports c1 == c2"

        self.fprs = DW_GCA(c1)

        self.add = shortcut

    def forward(self, x):
        y = self.fprs(x)
        return x + y if self.add else y

class CoDA(nn.Module):
    """
    CoDA（Cross-gated Dual-domain Adaptive Coordinate Attention）
    设计动机：
      1）坐标注意力对 H/W 方向进行解耦建模，但对不同噪声类型/强度的自适应性有限；
      2）仅依赖空间域统计，难以感知频域能量分布中与噪声相关的高频成分；
      3）H/W 两个分支彼此独立，缺乏跨方向的相互约束与选择。

    关键创新（相对原始 ACA 的“魔改”）：
      A. 频域门控（Frequency Gate）：使用 torch.fft.fft2 获取幅度谱，学习得到逐通道门控，
         在不显著增加开销的前提下引入“频域先验”，提升对不同噪声形态的适配性。
      B. 方向长程建模（Directional DW-Conv）：在 H/W 两个分支上加入深度可分一维卷积
         （k×1 与 1×k），抓取跨行/跨列的长程依赖，增强细粒度噪声的捕捉。
      C. 跨门控融合（Cross-gating）：使用对向分支的全局门信号对本分支进行调制，
         让 H/W 两个方向“相互选择、相互抑制”，避免冗余响应。
      D. 可学习温度与层缩放（Temperature & LayerScale）：为 Sigmoid 引入温度 τ（可学习），
         并对残差分支加入 γ 的层缩放，稳定训练、便于与现有骨干网即插即用。

    备注：保留了 ACA 的自适应缩放思想（alpha），并保持轻量化与即插即用特性。
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        alpha: float = 0.8,
        kernel_size: int = 7,
        use_frequency_gate: bool = True,
        layerscale_init: float = 0.05,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 需为奇数，便于对齐（padding 对称）"
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = max(8, in_channels // reduction)  # 瓶颈通道数
        self.alpha = alpha
        self.use_frequency_gate = use_frequency_gate

        # 共享 MLP：对 H/W 拼接后的方向统计做通道压缩与非线性映射
        # 微信公众号：CV缝合救星
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )

        # 方向长程建模：深度可分一维卷积（沿 H 与 W）
        # 目的：对每个方向引入更强的跨行/跨列上下文
        pad = kernel_size // 2
        self.dw_h = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            groups=self.mid_channels,
            bias=False,
        )
        self.dw_w = nn.Conv2d(
            self.mid_channels,
            self.mid_channels,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            groups=self.mid_channels,
            bias=False,
        )

        # 方向映射回原通道维度
        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, bias=False)

        # 频域门控：逐通道门（0~1），由幅度谱的全局统计产生
        if self.use_frequency_gate:
            self.freq_gate = nn.Sequential(
                nn.Conv2d(in_channels, max(8, in_channels // reduction), kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(8, in_channels // reduction), in_channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

        # 可学习温度，用于调节 Sigmoid 的锐度（τ 越小越“硬”）
        self.tau = nn.Parameter(torch.tensor(1.0))

        # 层缩放参数（LayerScale），稳定深层残差叠加
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1) * layerscale_init)

        # 轻微随机失活，进一步抑制过拟合（可选）
        self.dropout = nn.Dropout(p=0.05)

    def _sigmoid_temp(self, x):
        # 带温度的 Sigmoid：sigmoid(x / tau)
        return torch.sigmoid(x / (self.tau.abs() + 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, C, H, W]
        输出：
            y: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # -------- 1) 方向统计（与 ACA 一致的骨架）--------
        x_h = F.adaptive_avg_pool2d(x, (H, 1))  # [B, C, H, 1] —— 沿宽度聚合，保留行方向
        x_w = F.adaptive_avg_pool2d(x, (1, W))  # [B, C, 1, W] —— 沿高度聚合，保留列方向
        x_w = x_w.permute(0, 1, 3, 2)          # 变成 [B, C, W, 1]，便于与 H 方向拼接

        # 拼接后用共享 MLP 抽取方向相关的通道表征
        y = torch.cat([x_h, x_w], dim=2)       # [B, C, H+W, 1]
        y = self.shared_conv(y)                # [B, C', H+W, 1]，C' = mid_channels

        # 切分回两个方向分支，并做“方向深度可分一维卷积”的长程建模
        y_h, y_w = torch.split(y, [H, W], dim=2)   # y_h:[B,C',H,1], y_w:[B,C',W,1]
        y_w = y_w.permute(0, 1, 3, 2)              # y_w:[B,C',1,W]

        y_h = self.dw_h(y_h)                       # 在 H 方向做 k×1 的 DWConv
        y_w = self.dw_w(y_w)                       # 在 W 方向做 1×k 的 DWConv

        # 映射回原通道数，并应用自适应缩放 alpha（延续 ACA 思想）
        a_h_raw = self.conv_h(y_h) * self.alpha     # [B,C,H,1]
        a_w_raw = self.conv_w(y_w) * self.alpha     # [B,C,1,W]

        # -------- 2) 跨门控融合（Cross-gating）--------
        # 使用对向分支的全局门对本分支进行调制，增强“相互选择/抑制”
        gate_h_from_w = a_w_raw.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        gate_w_from_h = a_h_raw.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]

        a_h = self._sigmoid_temp(a_h_raw) * (1.0 + gate_h_from_w)   # [B,C,H,1]
        a_w = self._sigmoid_temp(a_w_raw) * (1.0 + gate_w_from_h)   # [B,C,1,W]

        # -------- 3) 频域门控（Dual-domain 之二：Frequency）--------
        if self.use_frequency_gate:
            # 计算幅度谱的均值特征（逐通道）
            # 注：仅用于产生门控标量，不做反变换，计算量可控
            Xf = torch.fft.fft2(x, norm='ortho')            # 复数张量
            mag = torch.abs(Xf)                             # 幅度谱 [B,C,H,W]
            mag_mean = mag.mean(dim=(2, 3), keepdim=True)   # [B,C,1,1]
            f_gate = self.freq_gate(mag_mean)               # [B,C,1,1] in [0,1]
        else:
            f_gate = 0.0

        # -------- 4) 融合与输出 --------
        # 方向注意力相加（与坐标注意力一致的融合策略），再乘以频域门控（1 + f_gate）
        attn = a_h + a_w                                   # [B,C,H,W]（自动广播）
        attn = attn * (1.0 + f_gate)                       # 双域融合：空间方向 + 频域门控
        attn = self.dropout(attn)

        # 残差式输出，LayerScale 提升稳定性
        out = x + self.gamma * (x * attn)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sa = self.sigmoid(self.conv(x))
        return x * sa


class MSGI_FCM(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        # 多尺度分支
        self.conv1 = Conv(dim, dim // 2, k=1)
        self.conv3 = Conv(dim, dim // 2, k=3)
        self.conv5 = Conv(dim, dim // 2, k=5)
        self.merge = Conv(dim // 2 * 3, dim, k=1)

        # 注意力模块
        self.channel_attn = ChannelAttention(dim)
        self.spatial_attn = SpatialAttention(dim)

        # 输出整合
        self.out_conv = Conv(dim, dim_out, k=1)
        self.residual = nn.Identity() if dim == dim_out else Conv(dim, dim_out, k=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)

        multi_scale = torch.cat([x1, x3, x5], dim=1)
        x_mixed = self.merge(multi_scale)

        x_c = self.channel_attn(x_mixed)
        x_s = self.spatial_attn(x_mixed)

        out = self.out_conv(x_c + x_s)
        return out + self.residual(x)


# ==============================
# 工具: 选择 GroupNorm 分组
# ==============================
def _choose_gn_groups(C: int) -> int:
    for g in [32, 16, 8, 4, 2, 1]:
        if C % g == 0:
            return g
    return 1


# ==============================
# 基础块: CLC (Conv-LeakyReLU-Conv)
# ==============================
class CLC(nn.Module):
    """
    CLCk: Conv k×k -> LeakyReLU -> Conv k×k
    """
    def __init__(self, in_ch, out_ch=None, k=3, negative_slope=0.1):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# 轻量上采样块: DNRU
# ==============================
class DNRU(nn.Module):
    """
    DNRU: 深度可分离卷积 3×3 + GroupNorm + ReLU + 可选上采样
    """
    def __init__(self, channels, up_scale=1):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.gn = nn.GroupNorm(_choose_gn_groups(channels), channels)
        self.relu = nn.ReLU(inplace=True)
        self.up_scale = up_scale

    def forward(self, x):
        x = self.dwconv(x)
        x = self.gn(x)
        x = self.relu(x)
        if self.up_scale and self.up_scale != 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x


# ==============================
# 通道向量 <-> 2D 网格 (用于 2D FFT)
# ==============================
def vector_to_grid(x_vec):
    """
    x_vec: (B, C, 1, 1) -> (B, 1, Hc, Wc), 同时返回 (Hc, Wc, C, pad)
    使 C 尽量铺成接近方形的网格以做 2D FFT
    """
    B, C, _, _ = x_vec.shape
    Hc = int(math.floor(math.sqrt(C)))
    Wc = int(math.ceil(C / Hc))
    pad = Hc * Wc - C
    if pad > 0:
        x_vec = F.pad(x_vec.view(B, C), (0, pad))  # 在通道描述末尾补零
        C_ = C + pad
    else:
        x_vec = x_vec.view(B, C)
        C_ = C
    grid = x_vec.view(B, 1, Hc, Wc)
    return grid, (Hc, Wc, C, pad)


def grid_to_vector(grid, meta):
    """
    grid: (B, 1, Hc, Wc) -> (B, C, 1, 1)
    """
    Hc, Wc, C, pad = meta
    B = grid.size(0)
    vec = grid.view(B, Hc * Wc)
    if pad > 0:
        vec = vec[:, :C]
    return vec.view(B, C, 1, 1)


# ==============================
# 频率半径网格 (0~1 归一化)
# ==============================
def normalized_freq_radius(h, w, device=None, dtype=None):
    """
    生成频率半径 r \in [0,1]，基于 torch.fft.fftfreq
    r = sqrt(fx^2 + fy^2) / r_max
    """
    fy = torch.fft.fftfreq(h, d=1.0).to(device=device, dtype=dtype)  # [-0.5,0.5) 尺度
    fx = torch.fft.fftfreq(w, d=1.0).to(device=device, dtype=dtype)
    fy, fx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(fx * fx + fy * fy)
    # 最大半径（Nyquist 对角）
    r_max = math.sqrt((0.5 ** 2) * 2.0)
    r = (r / r_max).clamp(0, 1)
    return r  # [H, W]


# ==============================
# ECA: 高效通道注意力（轻量1D卷积）
# ==============================
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # x: [B,C,H,W] -> [B,C,1,1] -> [B,1,C]
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)  # [B, C, 1] -> [B,1,C]
        y = self.conv1d(y).transpose(1, 2).unsqueeze(-1)             # [B,1,C] -> [B,C,1,1]
        return torch.sigmoid(y)


# ================================================
# FG-RCA: Fourier-Gated Residual Channel Attention
# ================================================
class FG_RCA(nn.Module):
    """
    🧠 模块名称：FG-RCA —— Fourier-Gated Residual Channel Attention
    设计要点：
    - Step1: GAP 得到通道描述向量
    - Step2: 向量 -> 2D 网格，做 FFT 分离振幅/相位
    - Step3: 对振幅/相位做非线性调制，并通过【频率门控】加强高频
    - Step4: IFFT -> 通道权重（Sigmoid）
    - Step5: 与 ECA 通道注意力融合，残差叠加；可选 DNRU 上采样
    """
    def __init__(
        self,
        channels: int,
        up_scale: int = 1,
        negative_slope: float = 0.1,
        eca_ksize: int = 3,
        freq_gate_init_t: float = 0.35,
        freq_gate_init_s: float = 8.0,
    ):
        super().__init__()
        self.channels = channels

        # 空间域轻量前处理（可学习更精细的低频/纹理前置特征）
        self.spatial_pre = CLC(channels, channels, k=3, negative_slope=negative_slope)

        # 频域 振幅/相位 轻量非线性映射
        act = nn.GELU()
        self.amp_mlp = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            act,
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
        )
        self.pha_mlp = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            act,
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
        )

        # 频率门控参数：阈值 t 与陡峭度 s（Sigmoid 门），以及高/低频整体缩放
        self.freq_t = nn.Parameter(torch.tensor(freq_gate_init_t, dtype=torch.float32))
        self.freq_s = nn.Parameter(torch.tensor(freq_gate_init_s, dtype=torch.float32))
        self.hi_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # 高频整体增益
        self.lo_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # 低频整体增益
        self.phs_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))  # 相位调制幅度

        # ECA 通道注意力
        self.eca = ECA(channels, k_size=eca_ksize)

        # 融合权重（可学习在频域与ECA之间的权衡）
        self.mix_alpha = nn.Parameter(torch.tensor(0.6, dtype=torch.float32))  # in [0,1] 约束用 sigmoid
        self.mix_beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))   # 额外尺度

        # 轻量后处理 + 可选上采样
        self.post = DNRU(channels, up_scale=up_scale)

    @staticmethod
    def _sigm(x):
        return torch.sigmoid(x)

    def _freq_gate(self, h, w, device, dtype):
        """
        基于半径频率 r 得到门控:
        gate = sigmoid( s * (r - t) )
        -> 高频区域 gate ~ 1，低频区域 gate ~ 0
        最终频域缩放: lo_gain*(1-gate) + hi_gain*gate
        """
        r = normalized_freq_radius(h, w, device=device, dtype=dtype)  # [H,W]
        gate = torch.sigmoid(self.freq_s * (r - self.freq_t))         # [H,W]
        scale = self.lo_gain * (1.0 - gate) + self.hi_gain * gate
        return gate, scale  # 两种形式: gate 用于调制相位，scale 用于调制振幅

    def forward(self, x):
        """
        x: [B,C,H,W] -> 输出同形状
        """
        B, C, H, W = x.shape
        assert C == self.channels, "channels mismatch"

        # 空间域预处理
        feat = self.spatial_pre(x)  # [B,C,H,W]

        # Step1: GAP -> 通道描述向量
        chan_desc = F.adaptive_avg_pool2d(feat, 1)  # [B,C,1,1]

        # Step2: 向量->网格，FFT 分离振幅/相位
        grid, meta = vector_to_grid(chan_desc)                          # [B,1,Hc,Wc]
        spec = torch.fft.fft2(grid)                                     # complex [B,1,Hc,Wc]
        amp = torch.abs(spec)                                           # [B,1,Hc,Wc]
        pha = torch.angle(spec)                                         # [B,1,Hc,Wc]

        # 频率门控 (依据网格大小计算)
        _, Hc, Wc = spec.shape[1], spec.shape[2], spec.shape[3]
        gate, scale = self._freq_gate(Hc, Wc, grid.device, grid.dtype)  # [Hc,Wc]

        # Step3: 振幅/相位非线性 + 频域门控
        # 振幅放大: amp' = amp * (1 + amp_mlp(amp)) * scale
        amp_adj = amp * (1.0 + self.amp_mlp(amp)) * scale.unsqueeze(0).unsqueeze(0)
        # 相位微调: pha' = pha + phs_scale * gate * tanh(pha_mlp(pha))
        pha_adj = pha + self.phs_scale * gate.unsqueeze(0).unsqueeze(0) * torch.tanh(self.pha_mlp(pha))

        # Step4: 频谱重建 -> IFFT -> 通道权重
        spec_new = torch.polar(amp_adj, pha_adj)                        # complex
        grid_ifft = torch.fft.ifft2(spec_new).real                      # [B,1,Hc,Wc]
        weight_vec = grid_to_vector(grid_ifft, meta)                    # [B,C,1,1]
        w_freq = torch.sigmoid(weight_vec)                              # 频域得到的通道权重

        # ECA 通道注意力
        w_eca = self.eca(feat)                                          # [B,C,1,1]

        # Step5: 融合与残差
        alpha = torch.sigmoid(self.mix_alpha)                            # [0,1]
        w = self._sigm(self.mix_beta) * (alpha * w_freq + (1 - alpha) * w_eca)
        y = feat * w                                                    # 通道注意力作用
        out = y + x                                                     # 残差

        # 轻量后处理（可选上采样）
        out = self.post(out)
        return out


class MiLKConvAttn(nn.Module):
    """
    MiLK-ConvAttn: Mixture-of-Large-Kernels Convolutional Attention
    - Long-range path: 从一个小型“大核字典”按样本×通道路由出 depthwise 大核，建模长程依赖
    - Local path: 动态 3x3 depthwise 卷积，捕获局部细节（实例敏感）
    - Bi-path gating: 两分支通道级软门控（softmax），自适应融合
    - Optional external path: 兼容外部共享大核 lk_filter（full conv），用于复现实验/对比
    """
    def __init__(
        self,
        pdim: int,
        proj_dim_in: Optional[int] = None,
        k: int = 13,             # 大核尺寸
        num_bases: int = 8,      # 大核字典规模
        sk_size: int = 3         # 动态小核尺寸
    ):
        super().__init__()
        self.pdim = pdim
        self.proj_dim_in = proj_dim_in if proj_dim_in is not None else pdim
        self.k = k
        self.num_bases = num_bases
        self.sk_size = sk_size

        hidden = max(16, self.proj_dim_in // 2)

        # ---- (1) 大核字典：共享、可学习、通道无关（depthwise 的核）----
        # 形状 [num_bases, 1, k, k]
        self.lk_bases = nn.Parameter(torch.randn(num_bases, 1, k, k) * 0.01)

        # ---- (2) 路由器：输出 [B, pdim, num_bases] 的权重，对字典做加权混合 ----
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, pdim * num_bases, 1)
        )

        # ---- (3) 动态 3×3 depthwise 小核（局部分支）----
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, pdim * self.sk_size * self.sk_size, 1)
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

        # ---- (4) 双路径软门控（每通道 2 个门控值，softmax）----
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.proj_dim_in, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, pdim * 2, 1)
        )

        # ---- (5) 融合投影（1×1），随后与残差相加 ----
        self.fuse = nn.Conv2d(pdim, pdim, 1)

    def forward(self, x: torch.Tensor, lk_filter: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, C, H, W]
        lk_filter (optional): [pdim, pdim, k, k] 的外部共享大核（full conv），作为额外路径
        """
        B, C, H, W = x.shape
        assert C >= self.pdim, "Input channels must be >= pdim"

        # 拆分注意力通道与旁路通道
        x1, x2 = x[:, :self.pdim], x[:, self.pdim:]

        # ---------- 长程分支：由字典路由得到每样本×每通道的大核 ----------
        # 路由权重 [B, pdim, num_bases]
        rw = self.router(x[:, :self.proj_dim_in]).view(B, self.pdim, self.num_bases)
        rw = torch.softmax(rw, dim=-1)

        # 合成 depthwise 大核 [B, pdim, 1, k, k]：对字典做加权求和
        # self.lk_bases: [num_bases, 1, k, k]
        composed_k = (rw[..., None, None, None] * self.lk_bases[None, None, ...]).sum(dim=2)

        # 使用合成大核做 depthwise 卷积（groups = B * pdim）
        x1_reshaped = rearrange(x1, 'b c h w -> 1 (b c) h w')
        composed_k_groups = rearrange(composed_k, 'b c o k1 k2 -> (b c) o k1 k2')
        out_lk = F.conv2d(x1_reshaped, composed_k_groups, padding=self.k // 2, groups=B * self.pdim)
        out_lk = rearrange(out_lk, '1 (b c) h w -> b c h w', b=B, c=self.pdim)

        # ---------- 局部分支：动态 3×3 depthwise ----------
        dyn_k = self.dwc_proj(x[:, :self.proj_dim_in]).view(B, self.pdim, 1, self.sk_size, self.sk_size)
        dyn_k = rearrange(dyn_k, 'b c o k1 k2 -> (b c) o k1 k2')
        out_dyn = F.conv2d(x1_reshaped, dyn_k, padding=self.sk_size // 2, groups=B * self.pdim)
        out_dyn = rearrange(out_dyn, '1 (b c) h w -> b c h w', b=B, c=self.pdim)

        # ---------- 可选外部 full-conv 路径（与原实现兼容） ----------
        if lk_filter is not None:
            out_ext = F.conv2d(x1, lk_filter, padding=lk_filter.shape[-1] // 2)
        else:
            out_ext = 0.0  # 标量 0，直接广播

        # ---------- 双路径软门控融合（外部路径并入长程分支） ----------
        g = self.gate(x[:, :self.proj_dim_in]).view(B, 2, self.pdim, 1, 1)
        g = torch.softmax(g, dim=1)
        g_lk, g_dyn = g[:, 0], g[:, 1]

        fused = g_lk * (out_lk + (out_ext if isinstance(out_ext, torch.Tensor) else 0.0)) + g_dyn * out_dyn

        # ---------- 1×1 融合投影 & 残差 ----------
        y1 = self.fuse(fused) + x1

        # 复原通道
        y = torch.cat([y1, x2], dim=1)
        return y

    def extra_repr(self):
        return f'pdim={self.pdim}, proj_dim_in={self.proj_dim_in}, k={self.k}, num_bases={self.num_bases}, sk_size={self.sk_size}'


class FreqAttnPlus(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(FreqAttnPlus, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.mid_dim = in_dim // 2

        self.reduce = nn.Conv2d(in_dim, self.mid_dim, 1)
        self.depthwise = nn.Conv2d(self.mid_dim, self.mid_dim, 3, padding=1, groups=self.mid_dim)
        self.norm = nn.BatchNorm2d(self.mid_dim)
        self.relu = nn.ReLU(inplace=True)

        # learnable gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.mid_dim, self.mid_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(self.mid_dim // 8, self.mid_dim, 1),
            nn.Sigmoid()
        )

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.post_proj = nn.Conv2d(self.mid_dim * 2, in_dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_r = self.relu(self.norm(self.depthwise(self.reduce(x))))  # B, C/2, H, W

        fft_feat = torch.fft.fft2(x_r.float(), norm='ortho')  # complex
        q = k = v = fft_feat

        # reshape into heads
        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = F.softmax(attn.real, dim=-1)  # 只用实部参与 softmax

        out = attn @ v.real
        out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h=self.num_heads, h1=H, w1=W)
        out = torch.fft.ifft2(out, norm='ortho').real  # 频域回到空间

        # 残差增强路径（门控频率残差）
        fwm = torch.fft.ifft2(fft_feat * torch.fft.fft2(self.gate(x_r) * x_r)).real
        fwm = torch.cat([out, fwm], dim=1)

        out = self.post_proj(fwm) + x  # 加残差
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class InterRowColSelfAttention(nn.Module):
    def __init__(self, in_dim, q_k_dim, patch_ini, axis='H'):
        """
        初始化方法，定义了卷积层和位置嵌入。
        Parameters:
        in_dim : int  # 输入张量的通道数
        q_k_dim : int  # Q 和 K 向量的通道数
        axis : str  # 注意力计算的轴 ('H', 'W')
        """
        super(InterRowColSelfAttention, self).__init__()
        self.in_dim = in_dim
        self.q_k_dim = q_k_dim
        self.axis = axis
        H, W = patch_ini[0], patch_ini[1]

        # 定义卷积层
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=q_k_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # 根据轴选择不同的位置信息嵌入
        if self.axis == 'H':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, H, 1))  # 高度方向嵌入
        elif self.axis == 'W':
            self.pos_embed = nn.Parameter(torch.zeros(1, q_k_dim, 1, W))  # 宽度方向嵌入
        else:
            raise ValueError("Axis must be one of 'H' or 'W'.")  # 如果轴不是 'H', 'W' 则报错

        # 使用 Xavier 初始化位置嵌入
        nn.init.xavier_uniform_(self.pos_embed)

        self.softmax = Softmax(dim=-1)  # 定义 softmax 层
        self.gamma = nn.Parameter(torch.zeros(1))  # 定义可训练的缩放参数
        self.ca = ChannelAttention(in_dim)

    def forward(self, x):
        """
        前向传播方法，计算注意力机制。
        参数：
        x : Tensor  # 输入的 4D 张量 (batch, channels, height, width)
        """
        B, C, H, W = x.size()

        # 计算 Q, K, V
        Q = self.query_conv(x) + self.pos_embed  # (B, q_k_dim, H, W) + pos_embed
        K = self.key_conv(x) + self.pos_embed  # (B, q_k_dim, H, W) + pos_embed
        V = self.value_conv(x)  # (B, in_dim, H, W)
        scale = math.sqrt(self.q_k_dim)  # 缩放因子

        # 根据注意力轴 ('H', 'W') 进行不同维度的处理
        if self.axis == 'H':  # 如果是高度方向
            Q = Q.permute(0, 2, 3, 1).contiguous()  # 重新排列维度为 (B, H, W, q_k_dim)
            Q = Q.view(B * W, H, self.q_k_dim)  # 展平为 (B*W, H, q_k_dim)

            K = K.permute(0, 2, 3, 1).contiguous()
            K = K.view(B * W, H, self.q_k_dim).permute(0, 2, 1).contiguous()  # 展平为 (B*W, q_k_dim, H)

            V = V.permute(0, 2, 3, 1).contiguous()
            V = V.view(B * W, H, self.in_dim)  # 展平为 (B*W, H, in_dim)

            attn = torch.bmm(Q, K) / scale  # 计算注意力矩阵 (B*W, H, H)
            attn = self.softmax(attn)  # 进行 softmax 操作

            out = torch.bmm(attn, V)  # 使用注意力矩阵加权 V (B*W, H, in_dim)
            out = out.view(B, W, H, self.in_dim).permute(0, 3, 2, 1).contiguous()  # 最终输出形状 (B, C, H, W)

        else:  # 如果是宽度方向
            Q = Q.permute(0, 2, 3, 1).contiguous()  # 重新排列维度为 (B, H, W, q_k_dim)
            Q = Q.view(B * H, W, self.q_k_dim)  # 展平为 (B*H, W, q_k_dim)

            K = K.permute(0, 2, 3, 1).contiguous()
            K = K.view(B * H, W, self.q_k_dim).permute(0, 2, 1).contiguous()  # 展平为 (B*H, q_k_dim, W)

            V = V.permute(0, 2, 3, 1).contiguous()
            V = V.view(B * H, W, self.in_dim)  # 展平为 (B*H, W, in_dim)

            attn = torch.bmm(Q, K) / scale  # 计算注意力矩阵 (B*H, W, W)
            attn = self.softmax(attn)  # 进行 softmax 操作

            out = torch.bmm(attn, V)  # 使用注意力矩阵加权 V (B*H, W, in_dim)
            out = out.view(B, H, W, self.in_dim).permute(0, 3, 1, 2).contiguous()  # 最终输出形状 (B, C, H, W)

        # 使用 gamma 融合输入和输出
        gamma = torch.sigmoid(self.gamma)
        out = gamma * out + (1 - gamma) * x  # 输出加权

        # 加入通道注意力
        ca_out = self.ca(out)
        out = out * ca_out

        return out

class EnhancedEMA(nn.Module):
    def __init__(self, channels, factor=8, dilation_rate=2):
        super(EnhancedEMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        # 原始 1x1 卷积，用于基本特征抽取
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)

        # 改进：使用扩展卷积代替原始的 3x3 卷积
        self.conv_dilated = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                      padding=dilation_rate, dilation=dilation_rate)

        # 新增的全局上下文卷积分支
        self.global_context_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. 生成全局上下文特征
        global_context = self.global_context_conv(self.agp(x))  # b, c, 1, 1

        # 2. 分组特征计算
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # 使用 1x1 卷积进行基础特征抽取
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 基于 1x1 和 Sigmoid 加权的特征
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # 改进：使用扩展卷积替代原始 3x3 卷积
        x2 = self.conv_dilated(group_x)

        # 生成加权注意力
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # 结合全局上下文特征，进行加权融合
        final_output = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        return final_output + global_context.expand_as(final_output)  # 加入全局上下文特征


class SelfAttentionSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        融合自注意力机制的SE模块。
        :param in_channels: 输入的通道数
        :param reduction: 压缩比例
        """
        super(SelfAttentionSEBlock, self).__init__()

        # SE模块
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 自注意力机制
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. 通道注意力机制
        avg_pool = self.global_avg_pool(x).view(b, c)  # 全局平均池化
        y = self.fc1(avg_pool)
        y = nn.ReLU()(y)
        y = self.fc2(y)
        y_channel = self.sigmoid(y).view(b, c, 1, 1)  # 通道权重

        # 2. 自注意力机制
        q = self.query(x).view(b, -1, h * w)  # [B, C//8, H*W]
        k = self.key(x).view(b, -1, h * w)  # [B, C//8, H*W]
        v = self.value(x).view(b, c, h * w)  # [B, C, H*W]

        attention = torch.bmm(q.permute(0, 2, 1), k)  # [B, H*W, H*W]
        attention = self.softmax(attention)  # 归一化
        attention = torch.bmm(v, attention.permute(0, 2, 1))  # [B, C, H*W]
        attention = attention.view(b, c, h, w)  # 恢复形状

        # 3. 融合通道和自注意力
        output = x * y_channel + attention  # 通道和自注意力加权融合
        return output

class FullyDynamicSEBlock(nn.Module):
    def __init__(self, in_channels, reduction_min=4, reduction_max=32):
        """
        完全动态通道压缩比例的SE模块。
        :param in_channels: 输入的通道数
        :param reduction_min: 最小压缩比例
        :param reduction_max: 最大压缩比例
        """
        super(FullyDynamicSEBlock, self).__init__()

        # 使用一个小型神经网络来预测每个通道的压缩比例
        self.fc1 = nn.Linear(in_channels, in_channels // 4)
        self.fc2 = nn.Linear(in_channels // 4, in_channels)  # 输出每个通道的压缩比例

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)  # 维度变换
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)  # 恢复维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活生成权重

        self.reduction_min = reduction_min  # 最小压缩比例
        self.reduction_max = reduction_max  # 最大压缩比例

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入的维度

        # 通过全局池化得到每个通道的全局信息
        avg_pool = self.global_avg_pool(x)  # 全局平均池化
        avg_pool = avg_pool.view(b, c)  # 展平以便输入到全连接层

        # 使用全连接层预测每个通道的压缩比例
        reduction_ratios = self.fc1(avg_pool)
        reduction_ratios = self.fc2(reduction_ratios)  # 获取每个通道的压缩比例
        reduction_ratios = torch.sigmoid(reduction_ratios)  # 使用sigmoid将其归一化到[0, 1]范围

        # 将压缩比例映射到 [reduction_min, reduction_max] 范围内
        reduction_ratios = reduction_ratios * (self.reduction_max - self.reduction_min) + self.reduction_min

        # 动态计算每个通道的压缩比例
        reduction_ratios = reduction_ratios.view(b, c, 1, 1)  # 调整为通道级别的压缩比例

        # 使用动态压缩比例进行通道压缩
        y = self.conv1(x)  # 对特征进行卷积变换
        y = self.relu(y)
        y = self.conv2(y)  # 恢复维度
        y = self.sigmoid(y)  # 生成注意力权重

        # 按照每个通道的动态压缩比例加权输入特征
        output = x * y * reduction_ratios  # 加权输入特征
        return output

# DynamicAttentionConv 模块，结合自适应卷积和注意力机制
class DynamicAttentionConv(nn.Module):
    def __init__(self, dim, k1, k2, attention_dim=32):
        super(DynamicAttentionConv, self).__init__()
        
        # 初始深度可分离卷积，增强局部特征学习
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        # 动态卷积，使用可调卷积核大小
        self.conv_spatial1 = nn.Conv2d(dim, dim, kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(dim, dim, kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
        
        # Attention机制，学习输入特征的重要性
        self.attn_conv = nn.Conv2d(dim, attention_dim, kernel_size=1)
        self.attn_activation = nn.Softmax(dim=1)
        
        # 最终输出卷积
        self.conv1 = nn.Conv2d(dim, dim, 1)
        
        # 添加一个卷积层，将 attn_weights 的通道数扩展为 dim
        self.attn_expand = nn.Conv2d(attention_dim, dim, kernel_size=1)

    def forward(self, x):
        # 初步卷积，提取特征
        attn = self.conv0(x)
        
        # 空间感知卷积，分别使用不同卷积核大小
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        
        # 自适应注意力，计算空间重要性
        attn_weights = self.attn_conv(attn)
        attn_weights = self.attn_activation(attn_weights)
        
        # 扩展 attn_weights 的通道数，以匹配 attn 的通道数
        attn_weights = self.attn_expand(attn_weights)
        
        # 将注意力加权到原始输入上
        attn = attn * attn_weights
        
        # 最终卷积层处理特征
        attn = self.conv1(attn)
        
        return attn

# DACModule模块，集成DynamicAttentionConv模块
class DACModule(nn.Module):
    def __init__(self, d_model, k1=1, k2=19):
        super(DACModule, self).__init__()
        
        # 两个1x1卷积层，用于投影和激活
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        
        # 引入DynamicAttentionConv模块作为核心
        self.dynamic_attention_unit = DynamicAttentionConv(d_model, k1, k2)
        
        # 另一个1x1卷积层
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        # 残差连接初始化
        shortcut = x.clone()
        
        # 通过第一个卷积层和激活函数
        x = self.proj_1(x)
        x = self.activation(x)
        
        # 通过核心的DynamicAttentionConv模块
        x = self.dynamic_attention_unit(x)
        
        # 通过第二个卷积层
        x = self.proj_2(x)
        
        # 加上残差连接，防止信息丢失
        x = x + shortcut
        return x

class SPRSA(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        """
        初始化SPR-SA模块
        
        dim: 输入特征的通道数
        growth_rate: 隐藏层通道数的增长倍率，默认为2.0
        """
        super(SPRSA, self).__init__()

        # 计算隐藏层通道数
        hidden_dim = int(dim * growth_rate)

        # 第一个卷积层：深度可分卷积（Depthwise Convolution），用于局部特征提取
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),  # 深度可分卷积，groups=dim表示每个通道独立卷积
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)  # 1x1卷积，用于通道间信息的融合
        )
        
        # 激活函数，使用GELU（Gaussian Error Linear Unit）
        self.act = nn.GELU()

        # 第二个卷积层，用于输出恢复到原始通道数
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        """
        前向传播

        x: 输入特征图，形状为 (batch_size, channels, height, width)
        返回：输出特征图，形状为 (batch_size, channels, height, width)
        """
        # 通过conv_0提取局部特征
        x = self.conv_0(x)
        
        # CV缝合救星：此时图像经过卷积后，需要进行空间上的调整，利用全局池化（global pooling）来处理局部信息
        x1 = F.adaptive_avg_pool2d(x, (1, 1))  # 自适应平均池化，将特征图池化为1x1
        x1 = F.softmax(x1, dim=1)  # 对通道维度进行softmax操作，类似于特征的注意力机制
        
        # 关键步骤：通过软加权对原始特征图进行调整，聚焦重要区域
        x = x1 * x  # 将权重应用到输入特征图
        
        # 激活函数：增加非线性
        x = self.act(x)
        
        # 通过conv_1恢复原始的通道数
        x = self.conv_1(x)
        
        return x

# -----------------------------
# 基础半卷积：对一半通道做3x3卷积，另一半通道原样保留
# -----------------------------
class HalfConv(nn.Module):
    def __init__(self, dim: int, n_div: int = 2, kernel_size: int = 3):
        """
        dim: 输入通道数
        n_div: 将通道切为 n_div 份，这里用 2 表示一半做卷积、一半不动
        kernel_size: 半卷积使用的卷积核尺寸（默认3）
        """
        super().__init__()
        assert n_div >= 2, "n_div 至少为 2 才有“半卷积”之意"
        self.dim_conv = dim // n_div               # 需要卷积的通道数
        self.dim_skip = dim - self.dim_conv        # 保留不变的通道数
        padding = kernel_size // 2                 # 保持特征图尺寸不变
        # 仅对前半部分做标准卷积（这里不使用BN，保持轻量与通用）
        self.partial_conv = nn.Conv2d(self.dim_conv, self.dim_conv,
                                      kernel_size=kernel_size, stride=1,
                                      padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # 按通道拆分为 [卷积部分, 跳过部分]
        x_conv, x_skip = torch.split(x, [self.dim_conv, self.dim_skip], dim=1)
        # 只对“卷积部分”做3x3卷积
        x_conv = self.partial_conv(x_conv)
        # 拼接回去，形成“半卷积”的输出
        out = torch.cat((x_conv, x_skip), dim=1)
        return out


# -----------------------------
# 工具函数：Channel Shuffle
# 作用：在组间打散通道，促进跨组的信息交互
# 若通道数不能被 groups 整除，则退化为不shuffle（保证鲁棒性）
# -----------------------------
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    b, c, h, w = x.shape
    if groups <= 1 or c % groups != 0:
        return x
    channels_per_group = c // groups
    # 形状变换：B, (g * cpg), H, W -> B, g, cpg, H, W
    x = x.view(b, groups, channels_per_group, h, w)
    # 交换组维与通道内维：B, cpg, g, H, W
    x = x.transpose(1, 2).contiguous()
    # 展平回原通道维度：B, C, H, W
    x = x.view(b, c, h, w)
    return x


# -----------------------------
# 轻量级 SE 门控（Squeeze-Excitation）
# 作用：利用全局上下文自适应地重标定各通道重要性
# -----------------------------
class SEGate(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        """
        dim: 通道数
        reduction: 降维比（越大代表越轻量）
        """
        super().__init__()
        hidden = max(dim // reduction, 4)  # 保底不小于4，避免极小通道数时退化
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，得到通道级描述
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)     # 较平滑的激活，训练更稳定
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w                          # 通道重标定


# -----------------------------
# 深度可分离卷积（Depthwise Separable）
# 作用：在保持计算量极低的前提下做空间/通道混合
# -----------------------------
class DWConvPW(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        """
        dim: 通道数（深度卷积和逐点卷积的输入/输出通道）
        kernel_size: 深度卷积核大小
        """
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                            stride=1, padding=padding, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()  # 轻量非线性提升表达

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(x)
        return x


# -----------------------------
# GS-HalfConv（Gated Shuffle Half-Convolution）主模块
# 设计要点：
# 1) 通道三分组：每组做 HalfConv（仅一半通道卷积）
# 2) Channel Shuffle：跨组打散通道，弥补“分组独立”的信息孤岛
# 3) 深度可分离卷积：轻量空间混合
# 4) SE 门控：全局自适应通道重标定
# 5) 残差 + 可学习缩放：稳定训练，便于堆叠
# -----------------------------
class GSHalfConv(nn.Module):
    def __init__(self, dim: int, groups: int = 3,
                 n_div: int = 2, se_reduction: int = 4,
                 shuffle_groups: int = 3, dw_kernel: int = 3):
        """
        dim: 输入/输出通道数
        groups: 通道分组数量（论文默认3组）
        n_div: HalfConv中“卷积份数”划分（默认2 -> 一半卷积）
        se_reduction: SE门控的降维比
        shuffle_groups: channel shuffle 的分组数（通常与 groups 相同）
        dw_kernel: 深度卷积核大小（3/5均可）
        """
        super().__init__()
        assert groups >= 1, "groups 至少为 1"

        # 计算每组通道数，余数并入最后一组，保证总通道数不变
        base = dim // groups
        remainder = dim - base * (groups - 1)
        group_dims = [base] * (groups - 1) + [remainder]

        # 为每一组构建 HalfConv
        self.group_blocks = nn.ModuleList([
            HalfConv(d, n_div=n_div, kernel_size=3) for d in group_dims
        ])

        # 记录一些元信息
        self.groups = groups
        self.shuffle_groups = shuffle_groups if (dim % shuffle_groups == 0) else 1

        # 轻量空间/通道混合：深度可分离卷积
        self.mix = DWConvPW(dim, kernel_size=dw_kernel)

        # 全局门控：SE
        self.se = SEGate(dim, reduction=se_reduction)

        # 残差缩放参数：初始化为一个较小值，有助于稳定训练
        self.res_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # 简单的归一化层（可选）：在轻量模型中常用 LayerNorm2d/BN
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # 1) 通道三分组：按计算好的每组通道数拆分
        #   注意：最后一组带上余数，兼容任意 dim
        splits = []
        start = 0
        for blk in self.group_blocks:
            d = blk.partial_conv.in_channels if isinstance(blk, HalfConv) else None  # 占位，无实际用
            # 由于上面的占位不可用，这里用更稳妥的做法：从模块参数反推通道数
            # HalfConv 无法直接拿到“全组通道数”，改为记录 split 大小：
            # 解决方案：重写 group_dims 存在模块中，或者在 ModuleList 外部保存
            # 为简洁，这里直接重新计算 group_dims
            pass

        # 为了简洁与鲁棒，这里重新计算 group_dims（与 __init__ 同逻辑）
        base = x.shape[1] // self.groups
        remainder = x.shape[1] - base * (self.groups - 1)
        group_dims = [base] * (self.groups - 1) + [remainder]

        xs = torch.split(x, group_dims, dim=1)

        # 2) 每组做 HalfConv（仅一半通道卷积）
        outs = []
        for t, blk in zip(xs, self.group_blocks):
            outs.append(blk(t))

        # 3) 组级拼接
        out = torch.cat(outs, dim=1)

        # 4) Channel Shuffle：跨组打散通道（若通道数不可整除则自动跳过）
        out = channel_shuffle(out, self.shuffle_groups)

        # 5) 轻量空间/通道混合（深度可分离卷积）
        out = self.mix(out)

        # 6) 全局通道门控（SE）
        out = self.se(out)

        # 7) 归一化 + 残差（带可学习缩放）
        out = self.norm(out)
        out = identity + self.res_scale * out
        return out


# -----------------------------
# 兼容你原始 CGHalfConv 的外层包装（可选）
# 若你希望直接替换 CGHalfConv，用下方类名即可
# -----------------------------
class CGHalfConv_GS(nn.Module):
    """
    用 GS-HalfConv 替换原 CGHalfConv 内部的三组 HalfConv，并额外引入
    shuffle + 深度可分离卷积 + SE 门控 + 残差缩放。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.block = GSHalfConv(dim=dim, groups=3, n_div=2,
                                se_reduction=4, shuffle_groups=3, dw_kernel=3)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class CoordAttMeanMax(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAttMeanMax, self).__init__()
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1_mean = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_mean = nn.BatchNorm2d(mip)
        self.conv2_mean = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.conv1_max = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1_max = nn.BatchNorm2d(mip)
        self.conv2_max = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Mean pooling branch
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.conv1_mean(y_mean)
        y_mean = self.bn1_mean(y_mean)
        y_mean = self.relu(y_mean)
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)

        # Max pooling branch
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.conv1_max(y_max)
        y_max = self.bn1_max(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)

        # Apply attention
        x_h_mean = self.conv2_mean(x_h_mean).sigmoid()
        x_w_mean = self.conv2_mean(x_w_mean).sigmoid()
        x_h_max = self.conv2_max(x_h_max).sigmoid()
        x_w_max = self.conv2_max(x_w_max).sigmoid()

        # Expand to original shape
        x_h_mean = x_h_mean.expand(-1, -1, h, w)
        x_w_mean = x_w_mean.expand(-1, -1, h, w)
        x_h_max = x_h_max.expand(-1, -1, h, w)
        x_w_max = x_w_max.expand(-1, -1, h, w)

        # Combine outputs
        attention_mean = identity * x_w_mean * x_h_mean
        attention_max = identity * x_w_max * x_h_max

        # Sum the attention outputs
        return attention_mean + attention_max

class MultiScaleFreqDenoise(nn.Module):
    """
    多尺度频域去噪模块，适用于大小不一、模糊、边缘不清晰的麻点
    """
    def __init__(self, channels, scales=[(0.05,0.15),(0.15,0.3),(0.3,0.45)]):
        """
        channels: 输入通道数
        scales: 多尺度高频抑制频率范围，每个 tuple 为 (低频比例, 高频比例)
        """
        super().__init__()
        self.channels = channels
        self.scales = scales
        # 为每个尺度生成可学习增益
        self.gains = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1, 1) * 0.5) for _ in scales])
        self.sigmoid = nn.Sigmoid()

        # 边缘增强分支
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.fft2(x)
        x_fft_shift = torch.fft.fftshift(x_fft)

        # 构建频率网格
        fy = torch.linspace(-0.5, 0.5, H, device=x.device)
        fx = torch.linspace(-0.5, 0.5, W, device=x.device)
        F_x, F_y = torch.meshgrid(fx, fy, indexing='xy')
        radius = torch.sqrt(F_x**2 + F_y**2).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        out_fft = x_fft_shift.clone()
        for scale, gain in zip(self.scales, self.gains):
            mask = torch.ones_like(radius)
            mask[(radius >= scale[0]) & (radius <= scale[1])] = 0.1
            mask = mask * self.sigmoid(gain)
            out_fft = out_fft * mask

        # IFFT 回空域
        out_fft = torch.fft.ifftshift(out_fft)
        x_out = torch.fft.ifft2(out_fft).real

        # 边缘增强残差分支
        edge = self.edge_conv(x - x_out)
        out = x_out + edge

        return out

# ----------------- 1. 基础算子层 -----------------

class SobelPrior(nn.Module):
    def __init__(self, in_channels=1):
        super(SobelPrior, self).__init__()
        self.in_channels = in_channels
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.repeat(in_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=self.in_channels)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)


# ----------------- 2. 核心创新组件 -----------------

class LocalDynamicEncoder(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        # 专门捕捉发亮长线的分支：使用 1x7 和 7x1 卷积（类似非对称卷积）
        self.line_h = nn.Conv2d(1, out_ch, kernel_size=(1, 7), stride=2, padding=(0, 3))
        self.line_v = nn.Conv2d(1, out_ch, kernel_size=(7, 1), stride=2, padding=(3, 0))
        
        # 专门捕捉发亮麻点的分支：标准 3x3
        self.point = nn.Conv2d(1, out_ch, 3, stride=2, padding=1)

        # ❌ 弃用 Global Average Pool
        # ✅ 使用局部空间注意力来决定权重
        self.spatial_att = nn.Sequential(
            nn.Conv2d(out_ch, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f_line = self.line_h(x) + self.line_v(x)
        f_point = self.point(x)
        
        # 动态融合：哪里像线，哪里就多用 f_line；哪里像点，就多用 f_point
        att = self.spatial_att(f_line + f_point)
        out = f_line * att + f_point * (1 - att)
        
        return torch.sigmoid(out)

class DefectPrior(nn.Module):
    """
    带软阈值去噪的多维先验提取
    """

    def __init__(self, in_ch=1):
        super().__init__()
        self.sobel = SobelPrior(in_channels=in_ch)

        # Laplacian Buffer 化，解决 WARNING
        lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('laplace_kernel', lap_k.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1))

        self.blur = nn.AvgPool2d(3, stride=1, padding=1)

        # 可学习的软阈值参数
        self.threshold = nn.Parameter(torch.ones(1) * 0.02)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.SiLU()
        )

    def soft_thresholding(self, x):
        # 抑制背景噪声，强化显著缺陷
        return torch.sign(x) * torch.relu(torch.abs(x) - self.threshold)

    def forward(self, x):
        # 1. 提取原始先验
        g = self.sobel(x)
        l = torch.abs(F.conv2d(x, self.laplace_kernel, padding=1, groups=x.shape[1]))
        c = torch.abs(x - self.blur(x))

        # 2. 软阈值纯化
        prior = torch.cat([self.soft_thresholding(g),
                           self.soft_thresholding(l),
                           self.soft_thresholding(c)], dim=1)

        return self.fuse(prior)


# ----------------- 3. 完整 Stem 模块 -----------------

class GTPStem(nn.Module):
    def __init__(self, in_ch=1, out_ch=64):
        super().__init__()

        # 先验分支
        self.prior_extractor = DefectPriorV3(in_ch)
        self.dynamic_encoder = LocalDynamicEncoder(out_ch)

        # 主干分支
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

        # 增强门控参数
        self.gain = nn.Parameter(torch.ones(1, out_ch, 1, 1) * 0.5)
        self.bias = nn.Parameter(torch.ones(1, out_ch, 1, 1) * -0.2)

    def forward(self, x):
        # 1. 主特征提取
        feat = self.stem(x)  # [B, C, H/2, W/2]

        # 2. 动态先验编码
        prior_map = self.prior_extractor(x)  # [B, 1, H, W]
        defect_gate = self.dynamic_encoder(prior_map)  # [B, C, H/2, W/2]

        # 3. 偏差感知门控增强
        # 只有当先验强度超过 bias 时，增强才显著生效
        active_gate = torch.relu(defect_gate + self.bias)
        out = feat * (1 + self.gain * active_gate)

        return out


    """
    Defect-Aware YOLO Stem
    """
    def __init__(self, in_ch=1, out_ch=64):
        super().__init__()

        self.prior = DefectPrior(in_ch)

        # 主干 stem（保持 YOLO 风格）
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

        # 缺陷先验编码
        self.prior_encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # 强化最强信号
            nn.Conv2d(1, out_ch, 1), # 调整通道
            nn.Sigmoid()
        )

    def forward(self, x):
        # 主特征
        feat = self.stem(x)  # [B,C,H/2,W/2]
        # print(feat.shape,x.shape)
        # 缺陷先验图
        defect_map = self.prior(x)  # [B,1,H,W]
        defect_gate = self.prior_encoder(defect_map)  # [B,C,H/2,W/2]

        # ✨ 关键：门控残差增强
        out = feat * (1 + defect_gate)

        return out
    
class SobelPrior(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        # 定义基础 Sobel 核
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # 🚀 修正：根据输入通道数重复权重，并使用 register_buffer
        self.register_buffer('sobel_x', kernel_x.repeat(in_channels, 1, 1, 1))
        self.register_buffer('sobel_y', kernel_y.repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        # 🚀 修正：groups 必须等于 self.in_channels
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=self.in_channels)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)


class LinePrior(nn.Module):

    def __init__(self, in_ch=1):
        super().__init__()
        # 1x7 和 7x1 卷积：专门抓水平和垂直细线
        self.conv_h = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.conv_v = nn.Conv2d(in_ch, in_ch, kernel_size=(7, 1), padding=(3, 0), bias=False)
        
        # 3x3 空洞卷积：增大感受野，捕捉不连续的划痕边缘
        self.conv_dilation = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=2, dilation=2, bias=False)

    def forward(self, x):
        # 提取不同方向的线性强度
        edge_h = self.conv_h(x)
        edge_v = self.conv_v(x)
        edge_d = self.conv_dilation(x)
        
        # 使用 abs 或 relu 提取高频线性分量
        line_feat = torch.abs(edge_h) + torch.abs(edge_v) + torch.abs(edge_d)
        return line_feat

class DefectPriorV3(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.in_ch = in_ch
        
        # 1. 梯度先验 (抓 Chipping/边缘)
        self.sobel = SobelPrior(in_channels=in_ch)
        
        # 2. 亮度与局部对比度先验 (抓 Splash/亮斑)
        self.blur_large = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.blur_small = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        
        # 3. 🚀 线性先验 (专攻 Scratch/划痕)
        self.line_detect = LinePrior(in_ch=in_ch)

        # 4. 融合层：输入通道变为 in_ch * 4
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 4, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x):
        # 分支1：梯度特征
        g = self.sobel(x)
        
        # 分支2：高光/显著特征
        h_light = torch.relu(x - self.blur_large(x))
        
        # 分支3：局部结构特征 (DoG)
        dog = torch.abs(self.blur_small(x) - self.blur_large(x))
        
        # 分支4：🚀 线性特征 (Line Detection)
        line = self.line_detect(x)

        # 拼接四个特征分支 [B, 4, H, W]
        combined = torch.cat([g, h_light, dog, line], dim=1)
        
        return self.fuse(combined)

class PriorBypass(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.extractor = DefectPriorV3(in_ch=in_ch)

    def forward(self, x):
        # 只输出一张原始尺寸或 P2 尺寸的先验图
        p_map = self.extractor(x)
        return p_map # [B, 1, H, W]