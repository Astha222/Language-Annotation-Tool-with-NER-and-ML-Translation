"""Export a Hugging Face token-classification (NER) model to ONNX and quantize it to INT8 for faster CPU inference.

    python export_onnx_ner.py dslim/bert-base-NER                       # writes models/bert-base-NER-onnx/
    python export_onnx_ner.py Jean-Baptiste/roberta-large-ner-english

Output folder (default backend/models/<name>-onnx/): model.onnx (fp32), model_int8.onnx (dynamic INT8 weights), plus the
tokenizer files and config.json that onnx_ner.py needs at run time. It then checks the INT8 model against the original
PyTorch model on a few sentences and prints how many predicted labels differ.

Running the finished model needs only `onnxruntime`. Exporting also needs the `onnx` package, which at the time of writing
requires protobuf >= 6.31 while many other packages still pin protobuf < 6. To keep the two apart, install onnx into its own
folder and point this script at it:

    pip install --target C:/path/to/onnx-tools --no-deps onnx "protobuf>=6.31.1" ml_dtypes
    set ONNX_TOOLS_DIR=C:/path/to/onnx-tools
"""
import argparse
import os
import sys
import time
from pathlib import Path

if os.getenv("ONNX_TOOLS_DIR"):
    sys.path.insert(0, os.environ["ONNX_TOOLS_DIR"])        # must happen before anything imports protobuf

HERE = Path(__file__).resolve().parent
CHECK_SENTENCES = [
    "Rahul Sharma works at Infosys in Pune.",
    "pleased to host #RUOK day with @ AmandaRishworth @ Parl House today",
    "The Borrower shall pay Barclays Bank PLC in London within thirty days.",
    "going to delhi with rahul tomorrow lol",
]


def export(model_id: str, out_dir: Path, opset: int = 17):
    import numpy as np
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id).eval()
    tok.save_pretrained(out_dir)
    model.config.save_pretrained(out_dir)

    enc = tok(["Rahul Sharma works at Infosys in Pune.", "Hello"], padding=True, return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in enc]
    fp32, int8 = out_dir / "model.onnx", out_dir / "model_int8.onnx"
    torch.onnx.export(model, tuple(enc[n] for n in names), str(fp32), input_names=names, output_names=["logits"],
                      dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in names}, "logits": {0: "batch", 1: "sequence"}},
                      opset_version=opset, dynamo=False)
    # per_channel=False: per-channel dynamic quantization has been reported to break some models completely
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=False)
    print(f"fp32 onnx {fp32.stat().st_size / 1e6:.0f} MB -> int8 onnx {int8.stat().st_size / 1e6:.0f} MB")

    # accuracy check against the original PyTorch model
    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    diff = total = 0
    for text in CHECK_SENTENCES:
        e = tok(text, return_tensors="pt")
        with torch.inference_mode():
            ref = model(**e).logits[0].argmax(-1).numpy()
        got = sess.run(None, {n: e[n].numpy() for n in names})[0][0].argmax(-1)
        diff += int((ref != got).sum())
        total += len(ref)
    print(f"INT8 vs PyTorch fp32: {diff} of {total} token labels differ on {len(CHECK_SENTENCES)} check sentences")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_id")
    ap.add_argument("--out", help="output folder (default: models/<name>-onnx)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else HERE / "models" / (args.model_id.split("/")[-1] + "-onnx")
    t0 = time.time()
    export(args.model_id, out)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
