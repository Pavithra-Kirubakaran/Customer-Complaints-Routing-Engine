from pathlib import Path
import traceback
import runpy


def main():
    model_dir = Path("models_v2/final_distilbert_model")
    output = model_dir / "model.onnx"
    try:
        env = runpy.run_path("scripts/convert_to_onnx.py")
        convert = env.get("convert")
        if not convert:
            raise RuntimeError("convert() not found in script")
        convert(model_dir, output)
        print("Conversion completed, wrote:", output)
    except Exception:
        print("Conversion failed with exception:")
        traceback.print_exc()


if __name__ == '__main__':
    main()
