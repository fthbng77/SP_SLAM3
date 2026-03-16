#!/usr/bin/env python3
"""
Export a place recognition model (NetVLAD or CosPlace) to TorchScript for SP-SLAM3.

Usage:
    pip install torch torchvision
    # For CosPlace:
    python scripts/export_place_recognition.py --model cosplace --output cosplace.pt
    # For NetVLAD:
    python scripts/export_place_recognition.py --model netvlad --output netvlad.pt

The exported model expects:
    Input:  image tensor [1, 3, 320, 320] (ImageNet normalized)
    Output: global descriptor [1, D] (D=512 for CosPlace, D=4096 for NetVLAD)
"""

import argparse
import torch
import torch.nn as nn


def export_cosplace(output_path, device):
    """Export CosPlace model."""
    try:
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone="ResNet18",
            fc_output_dim=512,
        )
    except Exception:
        print("Error: Could not load CosPlace model.")
        print("Install with: pip install torch torchvision")
        print("Requires internet access for torch.hub.load")
        return False

    model = model.to(device).eval()
    example = torch.randn(1, 3, 320, 320, device=device)

    try:
        traced = torch.jit.trace(model, example)
        traced.save(output_path)
        print(f"CosPlace model exported to: {output_path}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def export_netvlad(output_path, device):
    """Export NetVLAD model (using hloc implementation)."""
    try:
        from hloc.extractors.netvlad import NetVLAD
        model = NetVLAD()
    except ImportError:
        print("Error: hloc package not found for NetVLAD.")
        print("Install with: pip install hloc")
        print("Alternatively, use --model cosplace which has fewer dependencies.")
        return False

    model = model.to(device).eval()
    example = torch.randn(1, 3, 320, 320, device=device)

    try:
        traced = torch.jit.trace(model, example)
        traced.save(output_path)
        print(f"NetVLAD model exported to: {output_path}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export place recognition model to TorchScript")
    parser.add_argument("--model", type=str, default="cosplace",
                        choices=["cosplace", "netvlad"],
                        help="Model type (default: cosplace)")
    parser.add_argument("--output", type=str, default="cosplace.pt",
                        help="Output TorchScript model path")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Exporting {args.model} model (device: {args.device})...")

    if args.model == "cosplace":
        export_cosplace(args.output, args.device)
    elif args.model == "netvlad":
        export_netvlad(args.output, args.device)


if __name__ == "__main__":
    main()
