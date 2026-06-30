#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pidinet_edge.py  —  pretrained PiDiNet as a drop-in edge source for M3.

Replaces Canny entirely. Loads the model + checkpoint ONCE and returns a
full-resolution edge-PROBABILITY map in [0,1] for an RGB crop (the sigmoid is
already applied inside PiDiNet's forward). The probability map doubles as the
edge-STRENGTH signal in M3 (it is a far better strength than a Sobel magnitude).

Architecture is vendored in ./pidinet_pkg (the repo's own models package), the
default weights are ./table7_pidinet.pth (PiDiNet trained on BSDS500 + PASCAL
VOC — the strongest general edge model in the repo). PiDiNet uses the SAME
ImageNet normalization as DINOv3, so nothing new is introduced.

Attribution: PiDiNet — "Pixel Difference Networks for Efficient Edge Detection",
Su et al., ICCV 2021. github.com/hellozhuo/pidinet. License: MIT (research use).

Deps: torch, numpy, and the pidinet_pkg/ package + table7_pidinet.pth alongside.
"""
import argparse
import numpy as np
import torch

from pidinet_pkg.pidinet import pidinet

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class PiDiNetEdger:
    """Cached PiDiNet edge detector. Build once, call .prob(rgb) per crop."""

    def __init__(self, weights="table7_pidinet.pth", config="carv4",
                 sa=True, dil=True, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        args = argparse.Namespace(config=config, sa=sa, dil=dil)
        self.model = pidinet(args).eval().to(self.device)
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[pidinet] WARNING load: missing={len(missing)} "
                  f"unexpected={len(unexpected)} (architecture/config mismatch?)")
        else:
            print(f"[pidinet] table7 loaded clean on {self.device}")

    @torch.no_grad()
    def prob(self, rgb):
        """RGB uint8 HxWx3 -> edge-probability map HxW float in [0,1].
        Pads to a multiple of 16 (PiDiNet downsamples by 8) with edge-replication
        to limit the border seam, runs the fused side output, crops back."""
        h, w = rgb.shape[:2]
        H = ((h + 15) // 16) * 16
        W = ((w + 15) // 16) * 16
        pad = np.pad(rgb, ((0, H - h), (0, W - w), (0, 0)), mode="edge")
        x = ((pad.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
        out = self.model(torch.from_numpy(x).to(self.device))[-1]   # fused, sigmoid'd
        return out[0, 0].detach().cpu().numpy()[:h, :w]