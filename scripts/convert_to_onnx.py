"""Convert a Hugging Face local model (safetensors/pytorch) to ONNX.

Usage:
  python scripts/convert_to_onnx.py --model-dir models_v2/final_distilbert_model --output models_v2/final_distilbert_model/model.onnx
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def convert(model_dir: Path, output: Path, opset: int = 13, max_length: int = 256):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dummy = tokenizer(
        "This is a conversion sample.",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch"},
    }

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(output),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    model_dir = args.model_dir
    output = args.output or (model_dir / "model.onnx")
    convert(model_dir, output, opset=args.opset)
    print(f"Exported ONNX model to: {output}")


if __name__ == "__main__":
    main()
