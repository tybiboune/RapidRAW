"""Make the two models of the skin tool's "AI (learned)" retouch method, from Alibaba's ABPN skin retouching.

ABPN (Lei et al., "ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution
Photo", CVPR 2022; ModelScope damo/cv_unet_skin_retouching_torch, Apache-2.0) retouches skin the way a retoucher's
blend layers do, learned from retouched portraits:

  - abpn_retouch_blend.onnx, the "even skin" network (pytorch_model.pt, 'generator'): from the face cut out at
    512 x 512, a blend layer for each channel, 0.5 where nothing changes. The picture x (0..1) becomes
    (1 - 2m) x^2 + 2m x, a soft light: above 0.5 lightens, below darkens. The pipeline's own strength,
    `degree`, scales the layer around 0.5, so a Strength slider is exact and needs no second run.
  - abpn_retouch_inpaint.onnx, the "blemish" network (joint_20210926.pth, 'inpainting_net'): fills the places
    the blemish detector found (abpn_blemish_detection.onnx, made by export_blemish_onnx.py) with skin,
    a 576 x 576 tile at a time (512 and 32 of overlap each side), at the face's own resolution.

ModelScope publishes both as PyTorch weights only, so the networks are rebuilt here (the classes below are copied
from ModelScope, Apache-2.0, Copyright Alibaba) and exported once. Weights are read with weights_only=True: plain
tensors, no code. The files go in RapidRAW's models folder (on Windows %APPDATA%\\io.github.CyberTimon.RapidRAW\\models).

Needs: torch, onnx, onnxruntime, numpy.

    base=https://www.modelscope.cn/models/damo/cv_unet_skin_retouching_torch/resolve/master
    curl -L -o pytorch_model.pt $base/pytorch_model.pt
    curl -L -o joint_20210926.pth $base/joint_20210926.pth
    python export_abpn_retouch_onnx.py pytorch_model.pt joint_20210926.pth <models folder>

abpn_retouch_blend.onnx   in "image" 1x3x512x512 RGB scaled to -1..1; out "blend" 1x3x512x512, 0..1.
abpn_retouch_inpaint.onnx in "image" 1x3x576x576 RGB -1..1 with the blemishes zeroed, "keep" 1x1x576x576
                          (1 skin to keep, 0 to fill); out "filled" 1x3x576x576, -1..1.
"""

import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

BLEND_SIZE = 512
TILE = 512 + 2 * 32


# ---- the "even skin" network: a plain U-Net (ModelScope unet_deploy.py) ----------------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.mpconv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size()[2] - x1.size()[2]
        dx = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()
        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        # deep supervision heads, in the checkpoint but not used when exporting
        self.dsoutc4 = OutConv(256, n_classes)
        self.dsoutc3 = OutConv(128, n_classes)
        self.dsoutc2 = OutConv(64, n_classes)
        self.dsoutc1 = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))


# ---- the blemish filler: a gated-convolution U-Net (ModelScope inpainting_unet.py, gconv.py) --------------------


class GatedConvBNActiv(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, sample="none-3", activ="relu", bias=False):
        super().__init__()
        stride = 2 if sample == "down-3" else 1
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=bias)
        self.gate = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        images = self.conv(x)
        gates = torch.sigmoid(self.gate(x))
        if hasattr(self, "bn"):
            images = self.bn(images)
        if hasattr(self, "activation"):
            images = self.activation(images)
        return images * gates


class RetouchingNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.ec_images_1 = GatedConvBNActiv(in_channels, 64, bn=False, sample="down-3")
        self.ec_images_2 = GatedConvBNActiv(64, 128, sample="down-3")
        self.ec_images_3 = GatedConvBNActiv(128, 256, sample="down-3")
        self.ec_images_4 = GatedConvBNActiv(256, 512, sample="down-3")
        self.ec_images_5 = GatedConvBNActiv(512, 512, sample="down-3")
        self.ec_images_6 = GatedConvBNActiv(512, 512, sample="down-3")
        self.dc_images_6 = GatedConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_images_5 = GatedConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_images_4 = GatedConvBNActiv(512 + 256, 256, activ="leaky")
        self.dc_images_3 = GatedConvBNActiv(256 + 128, 128, activ="leaky")
        self.dc_images_2 = GatedConvBNActiv(128 + 64, 64, activ="leaky")
        self.dc_images_1 = GatedConvBNActiv(64 + in_channels, out_channels, bn=False, activ=None, bias=True)

    def forward(self, images, keep):
        skips = [torch.cat((images, keep), dim=1)]
        x = skips[0]
        for i in range(1, 7):
            x = getattr(self, f"ec_images_{i}")(x)
            skips.append(x)
        for i in range(6, 0, -1):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = torch.cat((x, skips[i - 1]), dim=1)
            x = getattr(self, f"dc_images_{i}")(x)
        return torch.tanh(x)


class Filled(nn.Module):
    """The filler with the pipeline's composite folded in: what is kept stays, the rest is the network's."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image, keep):
        kept = image * keep
        return kept + (1 - keep) * self.net(kept, keep)


def export(model, inputs, names, out_names, path):
    with torch.no_grad():
        reference = model(*inputs)
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=names,
        output_names=out_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = session.run(None, {n: t.numpy() for n, t in zip(names, inputs)})[0]
    print(f"{os.path.basename(path)}: max difference from PyTorch {np.abs(got - reference.numpy()).max():.2e}")


def main():
    generator_path, joint_path, out_dir = sys.argv[1:4]
    generator = UNet()
    state = torch.load(generator_path, map_location="cpu", weights_only=True)["generator"]
    generator.load_state_dict(state, strict=True)
    generator.eval()
    export(
        generator,
        (torch.rand(1, 3, BLEND_SIZE, BLEND_SIZE) * 2 - 1,),
        ["image"],
        ["blend"],
        os.path.join(out_dir, "abpn_retouch_blend.onnx"),
    )

    net = RetouchingNet()
    joint = torch.load(joint_path, map_location="cpu", weights_only=True)
    net.load_state_dict(joint["inpainting_net"], strict=True)
    net.eval()
    keep = (torch.rand(1, 1, TILE, TILE) > 0.05).float()
    export(
        Filled(net).eval(),
        (torch.rand(1, 3, TILE, TILE) * 2 - 1, keep),
        ["image", "keep"],
        ["filled"],
        os.path.join(out_dir, "abpn_retouch_inpaint.onnx"),
    )


if __name__ == "__main__":
    main()
