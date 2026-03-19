#!/usr/bin/env python3
"""
Export LightGlue model to TorchScript for C++ inference in SP-SLAM3.

Usage:
    pip install git+https://github.com/cvg/LightGlue.git
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
    print("Install with: pip install git+https://github.com/cvg/LightGlue.git")
    exit(1)


class LightGlueExport(nn.Module):
    """Simplified LightGlue wrapper for TorchScript export.

    Disables dynamic features (pruning, early stopping) that prevent
    TorchScript tracing, and provides a flat tensor interface compatible
    with the C++ LightGlue::match() method.
    """

    def __init__(self):
        super().__init__()
        # Disable dynamic features for traceability
        self.matcher = LightGlueOrig(features="superpoint", depth_confidence=-1, width_confidence=-1)
        self.matcher.eval()

    def forward(
        self,
        kpts0: torch.Tensor,    # [1, N, 2] normalized keypoints
        kpts1: torch.Tensor,    # [1, M, 2] normalized keypoints
        desc0: torch.Tensor,    # [1, N, 256] descriptors
        desc1: torch.Tensor,    # [1, M, 256] descriptors
    ):
        data0 = {
            "keypoints": kpts0,
            "descriptors": desc0,
        }
        data1 = {
            "keypoints": kpts1,
            "descriptors": desc1,
        }

        pred = self.matcher({"image0": data0, "image1": data1})

        # matches0: [1, N] where matches0[0, i] = index in image1 (-1 if unmatched)
        # matching_scores0: [1, N] confidence scores
        matches0 = pred["matches0"][0]           # [N]
        scores0 = pred["matching_scores0"][0]    # [N]

        # Filter valid matches and build [K, 2] pairs
        valid = matches0 > -1
        idx0 = torch.where(valid)[0]            # indices in image0
        idx1 = matches0[valid]                   # corresponding indices in image1
        match_scores = scores0[valid]            # confidence scores

        matches = torch.stack([idx0, idx1], dim=-1)  # [K, 2]

        return matches, match_scores


def main():
    parser = argparse.ArgumentParser(description="Export LightGlue to TorchScript")
    parser.add_argument("--output", type=str, default="lightglue.pt",
                        help="Output TorchScript model path")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Creating LightGlue wrapper (device: {args.device})...")
    model = LightGlueExport().to(args.device)
    model.eval()

    # Example inputs for tracing
    N, M = 200, 200
    example_kpts0 = torch.randn(1, N, 2, device=args.device)
    example_kpts1 = torch.randn(1, M, 2, device=args.device)
    example_desc0 = torch.randn(1, N, 256, device=args.device)
    example_desc1 = torch.randn(1, M, 256, device=args.device)

    # Warmup forward pass to verify
    print("Running warmup forward pass...")
    with torch.no_grad():
        matches, scores = model(example_kpts0, example_kpts1, example_desc0, example_desc1)
    print(f"  Warmup: {matches.shape[0]} matches found (shape: {matches.shape})")

    # Trace the model
    print("Tracing model with torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (example_kpts0, example_kpts1, example_desc0, example_desc1),
            strict=False,
        )

    traced.save(args.output)
    print(f"LightGlue model exported to: {args.output}")

    # Verify exported model
    print("Verifying exported model...")
    loaded = torch.jit.load(args.output, map_location=args.device)
    with torch.no_grad():
        m2, s2 = loaded(example_kpts0, example_kpts1, example_desc0, example_desc1)
    print(f"  Verification: {m2.shape[0]} matches, scores range [{s2.min():.3f}, {s2.max():.3f}]")
    print("Export successful!")


if __name__ == "__main__":
    main()
