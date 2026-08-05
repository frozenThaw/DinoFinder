from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel


class DinoClsEmbedding(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).last_hidden_state[:, 0, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DINOv2-small CLS embeddings to ONNX.")
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--output", type=Path, default=Path("models/dinov2_vits14.onnx"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = DinoClsEmbedding(AutoModel.from_pretrained(args.model).eval())
    sample = torch.randn(2, 3, 224, 224, dtype=torch.float32)
    with torch.inference_mode():
        reference = model(sample).cpu().numpy()

    torch.onnx.export(
        model,
        (sample,),
        args.output,
        input_names=["pixel_values"],
        output_names=["cls_embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "cls_embedding": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    try:
        import onnx

        onnx.checker.check_model(onnx.load(args.output))
    except ImportError:
        pass

    maximum_error = None
    cosine = None
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        actual = session.run(["cls_embedding"], {"pixel_values": sample.numpy()})[0]
        maximum_error = float(np.max(np.abs(reference - actual)))
        cosine = float(
            np.min(
                np.sum(reference * actual, axis=1)
                / (np.linalg.norm(reference, axis=1) * np.linalg.norm(actual, axis=1))
            )
        )
        if cosine < 0.999:
            raise RuntimeError(f"ONNX parity check failed: cosine={cosine}")
    except ImportError:
        pass

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "format_version": 1,
        "model": args.model,
        "sha256": digest,
        "input_name": "pixel_values",
        "output_name": "cls_embedding",
        "image_size": 224,
        "dimension": 384,
        "opset": 18,
        "dtype": "float32",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "validation_max_abs_error": maximum_error,
        "validation_min_cosine": cosine,
    }
    manifest_path = args.output.with_name("dinov2_vits14_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
