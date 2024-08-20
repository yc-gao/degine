#!/usr/bin/env python
import argparse

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    model = AlexNet()
    model = torch.jit.script(model)

    if args.output:
        model.save(args.output)


if __name__ == '__main__':
    main()
