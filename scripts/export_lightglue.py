#!/usr/bin/env python3
"""
Export LightGlue model to TorchScript for C++ inference in SP-SLAM3.

Usage:
    pip install lightglue
    python scripts/export_lightglue.py --output lightglue.pt

The exported model expects:
    Input:  kpts0 [1,N,2], kpts1 [1,M,2], desc0 [1,N,256], desc1 [1,M,256]
            (keypoints normalized to [-1, 1])
    Output: (matches [K,2], scores [K])
"""

import argparse
import torch
import torch.nn as nn

try:
    from lightglue import LightGlue as LightGlueOrig
except ImportError:
    print("Error: lightglue package not found.")
    print("Install with: pip install lightglue")
    exit(1)


class LightGlueWrapper(nn.Module):
    """Wrapper around LightGlue for TorchScript export."""

    def __init__(self):
        super().__init__()
        self.matcher = LightGlueOrig(features="superpoint")
        self.matcher.eval()

    def forward(
        self,
        kpts0: torch.Tensor,
        kpts1: torch.Tensor,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
    ):
        # Build the input dict that LightGlue expects
        data0 = {"keypoints": kpts0, "descriptors": desc0.transpose(-1, -2)}
        data1 = {"keypoints": kpts1, "descriptors": desc1.transpose(-1, -2)}

        pred = self.matcher({"image0": data0, "image1": data1})

        matches = pred["matches"]     # [K, 2]
        scores = pred["scores"]       # [K]

        return matches, scores


def main():
    parser = argparse.ArgumentParser(description="Export LightGlue to TorchScript")
    parser.add_argument("--output", type=str, default="lightglue.pt",
                        help="Output TorchScript model path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Creating LightGlue wrapper (device: {args.device})...")
    model = LightGlueWrapper().to(args.device)
    model.eval()

    # Trace with example inputs
    print("Tracing model...")
    N, M = 200, 200
    example_kpts0 = torch.randn(1, N, 2, device=args.device)
    example_kpts1 = torch.randn(1, M, 2, device=args.device)
    example_desc0 = torch.randn(1, N, 256, device=args.device)
    example_desc1 = torch.randn(1, M, 256, device=args.device)

    try:
        traced = torch.jit.script(model)
    except Exception:
        print("torch.jit.script failed, trying torch.jit.trace...")
        traced = torch.jit.trace(
            model,
            (example_kpts0, example_kpts1, example_desc0, example_desc1)
        )

    traced.save(args.output)
    print(f"LightGlue model exported to: {args.output}")


if __name__ == "__main__":
    main()
