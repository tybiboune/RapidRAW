"""Make the two deblurring models the Deblur touch-up tool reads, from their authors' PyTorch weights.

- motion / camera shake: NAFNet-GoPro-width32 (megvii-research/NAFNet, MIT)
- out of focus: Restormer single-image defocus deblurring (swz30/Restormer, MIT)

The networks are rebuilt by spandrel from the weights alone, and exported with the height and width
left free and named "h" and "w", which is what RapidRAW pins to its tile size. The files go in
RapidRAW's models folder (on Windows %APPDATA%\\io.github.CyberTimon.RapidRAW\\models).

Needs: torch, spandrel, spandrel_extra_arches, onnx, onnxruntime, numpy.

    curl -L -o NAFNet-GoPro-width32.pth \
        https://huggingface.co/mikestealth/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth
    curl -L -o restormer_defocus.pth \
        https://github.com/swz30/Restormer/releases/download/v1.0/single_image_defocus_deblurring.pth
    python export_deblur_onnx.py NAFNet-GoPro-width32.pth deblur_motion_nafnet.onnx
    python export_deblur_onnx.py restormer_defocus.pth deblur_defocus_restormer.onnx

In: "input", 1x3xHxW, RGB 0..1 (H and W multiples of 16). Out: "output", the same, deblurred.
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort
import spandrel_extra_arches
import torch
from spandrel import MAIN_REGISTRY, ModelLoader

MAIN_REGISTRY.add(*spandrel_extra_arches.EXTRA_REGISTRY)

weights, out = sys.argv[1:3]
# weights_only: the checkpoint is read as plain tensors, nothing in it is run
state = torch.load(weights, map_location="cpu", weights_only=True)
state = state.get("params", state)
model = ModelLoader().load_from_state_dict(state).model.eval()
print("architecture:", type(model).__name__)

dummy = torch.rand(1, 3, 256, 256)
with torch.no_grad():
    reference = model(dummy).numpy()
torch.onnx.export(
    model,
    dummy,
    out,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
    opset_version=17,
    dynamo=False,
)
onnx.checker.check_model(onnx.load(out))
session = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
got = session.run(None, {"input": dummy.numpy()})[0]
print("written", out, "- largest difference from PyTorch:", float(np.abs(got - reference).max()))
