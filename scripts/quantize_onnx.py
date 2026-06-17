"""Quantize an ONNX model using ONNX Runtime's dynamic quantization.

Usage:
  python scripts/quantize_onnx.py --input models_v2/final_distilbert_model/model.onnx --output models_v2/final_distilbert_model/model.quant.onnx
"""
import argparse
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    quantize_dynamic(str(args.input), str(args.output), weight_type=QuantType.QInt8)
    print(f"Quantized model written to: {args.output}")


if __name__ == "__main__":
    main()
