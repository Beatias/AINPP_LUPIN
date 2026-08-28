from __future__ import annotations

import logging
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ainpp_pb_latam.models.mfunet.blocks import DoubleConv, DownBlock, UpBlock

logger = logging.getLogger(__name__)


class MFUNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (64, 128, 256, 512, 1024),
        kernel_size: int = 3,
        mode: str = "motion_field",  # "regression" or "segmentation" or "motion_field"
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.mode = mode
        self.kernel_size = kernel_size
        self.bilinear = bilinear
        self._validate_cfg()

        # Adjustment to bilinear
        features = list(features)
        if bilinear:
            features[-1] = features[-1] // 2

        self.drop = nn.Dropout(p=0.5)

        self.stem = DoubleConv(in_channels, features[0], kernel_size=kernel_size)

        # Encoder
        self.down = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down.append(DownBlock(features[i], features[i + 1], kernel_size=kernel_size))

        # Decoder
        self.up = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.up.append(
                UpBlock(
                    features[i],
                    features[i - 1],
                    features[i - 1],
                    kernel_size=kernel_size,
                    bilinear=bilinear,
                )
            )

        if self.mode == "regression":
            self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)
            
        elif self.mode == "segmentation":
            self.head = nn.Sequential(
                nn.Conv2d(features[0], out_channels, kernel_size=1),
                nn.Sigmoid()
            )
            
        elif self.mode == "motion_field":
            # Padding "same" exige que o stride seja 1, o que é o padrão do Conv2d
            self.head = nn.Conv2d(features[0], 2, kernel_size=kernel_size, padding='same')
            
        else:
            raise NotImplementedError(f"Mode '{self.mode}' is not implemented.")

    def _validate_cfg(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0.")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be > 0.")
        if len(self.features) < 2:
            raise ValueError("features must have length >= 2.")
        if any(f <= 0 for f in self.features):
            raise ValueError("All feature sizes must be > 0.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not isinstance(self.bilinear, bool):
            raise ValueError("bilinear must be a boolean.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []

        x = self.stem(x)
        skips.append(x)

        for down_block in self.down:
            x = down_block(x)
            skips.append(x)

        # O último elemento é o bottleneck
        bottleneck = skips.pop() 
        
        # Aplicando o dropout no bottleneck (opcional, mas comum)
        x = self.drop(bottleneck)

        for i, up_block in enumerate(self.up):
            # skips agora contem apenas as conexões residuais, lidas de trás para frente
            skip = skips[-(i + 1)]
            x = up_block(x, skip)

        # Agora o head inclui as ativações/convs corretas de acordo com o `mode`
        return self.head(x)

    