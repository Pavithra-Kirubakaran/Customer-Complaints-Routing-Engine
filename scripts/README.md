Conversion and ONNX quantization scripts

Usage

- Convert a local HF model to ONNX:

```
python scripts/convert_to_onnx.py --model-dir models_v2/final_distilbert_model --output models_v2/final_distilbert_model/model.onnx
```

- Quantize the produced ONNX model (dynamic quantization):

```
python scripts/quantize_onnx.py --input models_v2/final_distilbert_model/model.onnx --output models_v2/final_distilbert_model/model.quant.onnx
```

Notes on Git history and pushing

- GitHub blocks files >100MB. After creating `model.onnx` / `model.quant.onnx`, remove the original large safetensors file from history or track it with Git LFS.

Remove cached file and push (simple):

```bash
git rm --cached models_v2/final_distilbert_model/model.safetensors
git commit -m "remove large safetensors file"
git push origin main
```

To keep large files tracked going forward, use Git LFS:

```bash
git lfs install
git lfs track "models_v2/final_distilbert_model/*.safetensors"
git add .gitattributes
git add models_v2/final_distilbert_model/model.safetensors
git commit -m "track model with Git LFS"
git push origin main
```
