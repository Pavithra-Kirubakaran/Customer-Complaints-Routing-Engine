from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXSequenceClassifier:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.onnx_path = self.model_dir / "model.quant.onnx"
        if not self.onnx_path.exists():
            self.onnx_path = self.model_dir / "model.onnx"

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found in {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)

        config_path = self.model_dir / "config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            self.id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
        else:
            self.id2label = None

        # classes_ list in label order (0..N-1)
        if self.id2label:
            self.classes_ = [self.id2label[i] for i in sorted(self.id2label.keys())]
        else:
            # unknown number of classes until inference; set empty and fill on first run
            self.classes_ = []

        self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        toks = self.tokenizer(texts, truncation=True, padding=True, return_tensors="np")
        input_feed = {
            "input_ids": toks["input_ids"].astype(np.int64),
            "attention_mask": toks["attention_mask"].astype(np.int64),
        }

        outputs = self.session.run(None, input_feed)
        logits = outputs[0]
        # softmax
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        pred_ids = np.argmax(probs, axis=1)

        # build classes_ if missing
        if not self.classes_:
            num_classes = logits.shape[1]
            if self.id2label:
                self.classes_ = [self.id2label.get(i, str(i)) for i in range(num_classes)]
            else:
                self.classes_ = [str(i) for i in range(num_classes)]

        labels = []
        for i, pid in enumerate(pred_ids):
            label = self.classes_[int(pid)]
            labels.append((label, float(probs[i, pid])))
        return labels

    def predict_proba(self, texts):
        """Return probability matrix shape (n_samples, n_classes)."""
        if isinstance(texts, str):
            texts = [texts]

        toks = self.tokenizer(texts, truncation=True, padding=True, return_tensors="np")
        input_feed = {
            "input_ids": toks["input_ids"].astype(np.int64),
            "attention_mask": toks["attention_mask"].astype(np.int64),
        }
        outputs = self.session.run(None, input_feed)
        logits = outputs[0]
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        # ensure classes_
        if not self.classes_:
            num_classes = logits.shape[1]
            if self.id2label:
                self.classes_ = [self.id2label.get(i, str(i)) for i in range(num_classes)]
            else:
                self.classes_ = [str(i) for i in range(num_classes)]

        return probs


__all__ = ["ONNXSequenceClassifier"]
