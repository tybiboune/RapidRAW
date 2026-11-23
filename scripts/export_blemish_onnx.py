"""Make abpn_blemish_detection.onnx, the model the skin tool's "Find spots with AI" switch reads.

It is the blemish detector of Alibaba's ABPN skin retouching (CVPR 2022, ModelScope
damo/cv_unet_skin_retouching_torch, Apache-2.0): a U-Net trained on portraits to find spots, moles,
milia and acne marks. ModelScope publishes it as PyTorch weights only, inside the pipeline's joint
checkpoint, so this rebuilds the network (the two classes below are copied from ModelScope, Apache-2.0,
Copyright Alibaba) and exports it once with the sigmoid folded in. The file goes in RapidRAW's models
folder (on Windows %APPDATA%\\io.github.CyberTimon.RapidRAW\\models).

Needs: torch, onnx, onnxruntime, numpy.

    curl -L -o joint_20210926.pth \
        https://www.modelscope.cn/models/damo/cv_unet_skin_retouching_torch/resolve/master/joint_20210926.pth
    python export_blemish_onnx.py joint_20210926.pth abpn_blemish_detection.onnx

In: "image", 1x3x768x768, RGB scaled to -1..1, a square crop of 1.5 times the face box around it.
Out: "blemishes", 1x1x768x768, how likely each pixel is to be part of a blemish, 0 to 1.
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 768


class ConvBNActiv(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, sample="none-3", activ="relu"):
        super().__init__()
        stride, padding = (2, 1) if sample == "down-3" else (1, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=False)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, "bn"):
            x = self.bn(x)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x


class DetectionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.ec_images_1 = ConvBNActiv(n_channels, 64, bn=False, sample="down-3")
        self.ec_images_2 = ConvBNActiv(64, 128, sample="down-3")
        self.ec_images_3 = ConvBNActiv(128, 256, sample="down-3")
        self.ec_images_4 = ConvBNActiv(256, 512, sample="down-3")
        self.ec_images_5 = ConvBNActiv(512, 512, sample="down-3")
        self.ec_images_6 = ConvBNActiv(512, 512, sample="down-3")
        self.dc_images_6 = ConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_images_5 = ConvBNActiv(512 + 512, 512, activ="leaky")
        self.dc_images_4 = ConvBNActiv(512 + 256, 256, activ="leaky")
        self.dc_images_3 = ConvBNActiv(256 + 128, 128, activ="leaky")
        self.dc_images_2 = ConvBNActiv(128 + 64, 64, activ="leaky")
        self.dc_images_1 = nn.Conv2d(64 + n_channels, n_classes, kernel_size=1)

    def forward(self, x):
        skips = [x]
        for i in range(1, 7):
            skips.append(getattr(self, f"ec_images_{i}")(skips[-1]))
        logits = skips[6]
        for i in range(6, 0, -1):
            logits = F.interpolate(logits, scale_factor=2, mode="nearest")
            logits = torch.cat((logits, skips[i - 1]), dim=1)
            logits = getattr(self, f"dc_images_{i}")(logits)
        return torch.sigmoid(logits)


weights, out = sys.argv[1:3]
# weights_only: the checkpoint is read as plain tensors, nothing in it is run
checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
state = checkpoint["detection_net"]
state = {k.removeprefix("module."): v for k, v in state.items()}
net = DetectionUNet()
net.load_state_dict(state, strict=True)
net.eval()

dummy = torch.rand(1, 3, SIZE, SIZE) * 2 - 1
with torch.no_grad():
    reference = net(dummy).numpy()
torch.onnx.export(
    net,
    dummy,
    out,
    input_names=["image"],
    output_names=["blemishes"],
    opset_version=17,
    dynamo=False,
)
onnx.checker.check_model(onnx.load(out))
got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"image": dummy.numpy()})[0]
print("written", out, "- largest difference from PyTorch:", float(np.abs(got - reference).max()))
