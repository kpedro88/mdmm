#!/usr/bin/env python3

import argparse

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

import mdmm


class TVLoss(nn.Module):
    def forward(self, input):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
        y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
        diff = x_diff**2 + y_diff**2 + 1e-8
        return diff.sum(dim=1).sqrt().sum()


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('input_image', type=str,
                   help='the input image')
    p.add_argument('output_image', type=str, nargs='?', default='out.png',
                   help='the output image')
    p.add_argument('--max-tv', type=float, default=0.02,
                   help='the maximum allowable total variation per sample')
    p.add_argument('--damping', type=float, default=1e-2,
                   help='the constraint damping strength')
    p.add_argument('--lr', type=float, default=2e-3,
                   help='the learning rate')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    pil_image = Image.open(args.input_image).resize((128, 128), Image.LANCZOS)
    target = TF.to_tensor(pil_image)[None].to(device)
    input = target.clone().requires_grad_()
    # torch.manual_seed(0)
    # target += torch.randn_like(target) / 10
    # target.clamp_(0, 1)

    crit_l2 = nn.MSELoss(reduction='sum')
    crit_tv = TVLoss()
    max_tv = args.max_tv * input.numel()

    mdmm_mod = mdmm.MDMM([mdmm.MaxConstraint(lambda: crit_tv(input), max_tv, args.damping)])
    opt = mdmm_mod.make_optimizer([input], lr=args.lr)

    try:
        i = 0
        while True:
            i += 1
            loss = crit_l2(input, target)
            lagrangian, losses = mdmm_mod(loss)
            msg = '{} l2={:g}, tv={:g}'
            print(msg.format(i,
                             loss.item() / input.numel(),
                             losses[0].item() / input.numel()))
            if not lagrangian.isfinite():
                break
            opt.zero_grad()
            lagrangian.backward()
            opt.step()
    except KeyboardInterrupt:
        pass

    TF.to_pil_image(input[0].clamp(0, 1)).save(args.output_image)


if __name__ == '__main__':
    main()