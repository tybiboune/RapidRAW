"""Make ffhq_wrinkle_unet.onnx, the model the skin tool's "Find lines with AI" switch reads.

The FFHQ-Wrinkle U-Net (ResNet50 encoder, https://huggingface.co/rmsandu/ffhq-wrinkle-unet) is
published as PyTorch weights only, under CC BY-NC-SA 4.0, so RapidRAW does not download it: this
exports it once, with the sigmoid folded in, and the file goes in RapidRAW's models folder
(on Windows %APPDATA%\\io.github.CyberTimon.RapidRAW\\models).

Needs: torch, torchvision, safetensors, onnx, onnxruntime, numpy.

    git clone https://github.com/rmsandu/FFHQ-detect-face-wrinkles
    curl -L -o wrinkle_model.safetensors \
        https://huggingface.co/rmsandu/ffhq-wrinkle-unet/resolve/main/wrinkle_model.safetensors
    python export_wrinkle_onnx.py FFHQ-detect-face-wrinkles wrinkle_model.safetensors ffhq_wrinkle_unet.onnx

In: "image", 1x3x512x512, ImageNet-normalised RGB of an FFHQ-aligned face.
Out: "wrinkles", 1x1x512x512, how likely each pixel is to be a wrinkle, 0 to 1.
"""

import sys

repo, weights, out = sys.argv[1:4]
sys.path.insert(0, repo)

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from unet import UNet  # noqa: E402


class WithSigmoid(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


net = UNet(n_channels=3, n_classes=1, bilinear=False, pretrained=False, freeze_encoder=True)
net.load_state_dict(load_file(weights, device="cpu"), strict=True)
model = WithSigmoid(net.eval()).eval()

dummy = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    reference = model(dummy).numpy()

torch.onnx.export(
    model,
    dummy,
    out,
    input_names=["image"],
    output_names=["wrinkles"],
    opset_version=17,
    dynamo=False,
)

onnx.checker.check_model(onnx.load(out))
got = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(None, {"image": dummy.numpy()})[0]
print("written", out, "- largest difference from PyTorch:", float(np.abs(got - reference).max()))
