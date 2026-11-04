from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import numpy as np
import onnxruntime as ort


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from similar_finder import (  # noqa: E402
    DINOV2_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SimilarFinderError,
    discover_images,
    load_image_variants,
    source_fingerprint,
)


CACHE_VERSION = 1
MODEL_NAME = "local:models/dinov2_vits14.onnx"


@dataclass(frozen=True)
class VectorEntry:
    path: str
    variant: str


@dataclass(frozen=True)
class SimilarPair:
    first: str
    second: str
    cosine_similarity: float
    first_variant: str
    second_variant: str


@dataclass(frozen=True)
class SimilarGroup:
    paths: list[str]
    pair_count: int
    minimum_similarity: float
    maximum_similarity: float


class OnnxDinoV2Embedder:
    def __init__(self, model_path: Path, manifest_path: Path, batch_size: int) -> None:
        if batch_size < 1:
            raise SimilarFinderError("batch-size must be greater than zero")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual_hash != manifest["sha256"]:
            raise SimilarFinderError("DINOv2 model checksum mismatch")

        self.batch_size = batch_size
        self.input_name = str(manifest["input_name"])
        self.output_name = str(manifest["output_name"])
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def embed(self, images: Sequence[Any]) -> np.ndarray:
        from PIL import Image

        if not images:
            return np.empty((0, 384), dtype=np.float32)

        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_STD, dtype=np.float32)
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            tensors: list[np.ndarray] = []
            for image in batch:
                resized = image.convert("RGB").resize(
                    (DINOV2_IMAGE_SIZE, DINOV2_IMAGE_SIZE), Image.Resampling.BICUBIC
                )
                pixels = np.asarray(resized, dtype=np.float32) / 255.0
                pixels = (pixels - mean) / std
                tensors.append(pixels.transpose(2, 0, 1))
            output = self.session.run(
                [self.output_name],
                {self.input_name: np.stack(tensors)},
            )[0].astype(np.float32, copy=False)
            norms = np.linalg.norm(output, axis=1, keepdims=True)
            output /= np.maximum(norms, np.finfo(np.float32).eps)
            outputs.append(output)
        return np.concatenate(outputs, axis=0)


def cache_fingerprint(root: Path, paths: Sequence[Path], model_path: Path) -> str:
    base = source_fingerprint(root, paths, MODEL_NAME)
    digest = hashlib.sha256()
    digest.update(f"{CACHE_VERSION}:{base}:".encode("utf-8"))
    digest.update(hashlib.sha256(model_path.read_bytes()).digest())
    return digest.hexdigest()


def extract_vectors(
    root: Path,
    paths: Sequence[Path],
    embedder: OnnxDinoV2Embedder,
    progress_every: int,
) -> tuple[np.ndarray, list[VectorEntry], list[str]]:
    vectors: list[np.ndarray] = []
    entries: list[VectorEntry] = []
    skipped: list[str] = []
    pending_images: list[Any] = []
    pending_entries: list[VectorEntry] = []
    flush_limit = max(embedder.batch_size * 4, embedder.batch_size)

    def flush() -> None:
        if not pending_images:
            return
        vectors.append(embedder.embed(pending_images))
        entries.extend(pending_entries)
        for image in pending_images:
            image.close()
        pending_images.clear()
        pending_entries.clear()

    for number, path in enumerate(paths, start=1):
        try:
            variants = load_image_variants(path)
        except SimilarFinderError as exc:
            skipped.append(str(exc))
            continue
        relative = path.relative_to(root).as_posix()
        for variant, image in variants:
            pending_images.append(image)
            pending_entries.append(VectorEntry(relative, variant))
        if len(pending_images) >= flush_limit:
            flush()
        if number % progress_every == 0 or number == len(paths):
            print(f"embedded {number}/{len(paths)} images", flush=True)
    flush()

    if not vectors:
        raise SimilarFinderError("no decodable PNG images were found")
    return np.concatenate(vectors, axis=0), entries, skipped


def load_or_build_cache(
    root: Path,
    paths: Sequence[Path],
    cache_dir: Path,
    model_path: Path,
    embedder: OnnxDinoV2Embedder,
    rebuild: bool,
    progress_every: int,
) -> tuple[np.ndarray, list[VectorEntry], list[str]]:
    fingerprint = cache_fingerprint(root, paths, model_path)
    vectors_path = cache_dir / "vectors.npy"
    metadata_path = cache_dir / "metadata.json"
    if not rebuild and vectors_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["fingerprint"] == fingerprint:
                entries = [VectorEntry(**item) for item in metadata["entries"]]
                vectors = np.load(vectors_path)
                if vectors.shape == (len(entries), 384):
                    print(f"reused {len(entries)} cached DINOv2 vectors", flush=True)
                    return vectors, entries, list(metadata.get("skipped", []))
        except (OSError, ValueError, KeyError, TypeError):
            pass

    vectors, entries, skipped = extract_vectors(
        root, paths, embedder, progress_every
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_vectors = vectors_path.with_suffix(".npy.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    with temporary_vectors.open("wb") as file:
        np.save(file, vectors)
    temporary_metadata.write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint,
                "image_count": len(paths) - len(skipped),
                "vector_count": len(entries),
                "entries": [asdict(entry) for entry in entries],
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_vectors, vectors_path)
    os.replace(temporary_metadata, metadata_path)
    return vectors, entries, skipped


def alpha_signature(path: Path) -> str:
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
    alpha = image.getchannel("A")
    digest = hashlib.sha256()
    digest.update(image.width.to_bytes(8, "little"))
    digest.update(image.height.to_bytes(8, "little"))
    digest.update(alpha.tobytes())
    return digest.hexdigest()


def build_alpha_signatures(
    root: Path,
    entries: Sequence[VectorEntry],
    progress_every: int = 1000,
) -> dict[str, str]:
    paths = sorted({entry.path for entry in entries}, key=str.casefold)
    signatures: dict[str, str] = {}
    for number, relative_path in enumerate(paths, start=1):
        signatures[relative_path] = alpha_signature(root / relative_path)
        if number % progress_every == 0 or number == len(paths):
            print(f"hashed alpha {number}/{len(paths)} images", flush=True)
    return signatures


def find_pairs(
    vectors: np.ndarray,
    entries: Sequence[VectorEntry],
    threshold: float,
    alpha_signatures: Mapping[str, str] | None = None,
) -> list[SimilarPair]:
    try:
        import faiss
    except ImportError as exc:
        raise SimilarFinderError("faiss-cpu is required") from exc

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    search_threshold = (
        float(np.nextafter(np.float32(1.0), np.float32(-np.inf)))
        if threshold == 1.0
        else threshold
    )
    limits, scores, indices = index.range_search(vectors, search_threshold)

    best: dict[tuple[str, str], SimilarPair] = {}
    for first_index, first_entry in enumerate(entries):
        start = int(limits[first_index])
        end = int(limits[first_index + 1])
        for offset in range(start, end):
            second_index = int(indices[offset])
            if second_index <= first_index:
                continue
            second_entry = entries[second_index]
            if first_entry.path == second_entry.path:
                continue
            if (
                alpha_signatures is not None
                and alpha_signatures[first_entry.path] != alpha_signatures[second_entry.path]
            ):
                continue
            if first_entry.path.casefold() <= second_entry.path.casefold():
                first, second = first_entry, second_entry
            else:
                first, second = second_entry, first_entry
            key = (first.path, second.path)
            score = float(np.clip(scores[offset], -1.0, 1.0))
            if score < threshold:
                continue
            candidate = SimilarPair(
                first.path,
                second.path,
                score,
                first.variant,
                second.variant,
            )
            current = best.get(key)
            if current is None or candidate.cosine_similarity > current.cosine_similarity:
                best[key] = candidate
    return sorted(
        best.values(),
        key=lambda pair: (-pair.cosine_similarity, pair.first.casefold(), pair.second.casefold()),
    )


def build_groups(pairs: Sequence[SimilarPair]) -> list[SimilarGroup]:
    parent: dict[str, str] = {}
    sizes: dict[str, int] = {}

    def find(path: str) -> str:
        parent.setdefault(path, path)
        current = path
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = parent[current]
        return current

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if sizes.get(first_root, 1) < sizes.get(second_root, 1):
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        sizes[first_root] = sizes.get(first_root, 1) + sizes.get(second_root, 1)

    for pair in pairs:
        union(pair.first, pair.second)

    paths_by_root: dict[str, list[str]] = {}
    for path in parent:
        paths_by_root.setdefault(find(path), []).append(path)
    pairs_by_root: dict[str, list[SimilarPair]] = {}
    for pair in pairs:
        pairs_by_root.setdefault(find(pair.first), []).append(pair)

    groups: list[SimilarGroup] = []
    for root, paths in paths_by_root.items():
        group_pairs = pairs_by_root[root]
        scores = [pair.cosine_similarity for pair in group_pairs]
        groups.append(
            SimilarGroup(
                paths=sorted(paths, key=str.casefold),
                pair_count=len(group_pairs),
                minimum_similarity=min(scores),
                maximum_similarity=max(scores),
            )
        )
    return sorted(
        groups,
        key=lambda group: (-len(group.paths), -group.maximum_similarity, group.paths[0].casefold()),
    )


def write_results(
    output_dir: Path,
    root: Path,
    pairs: Sequence[SimilarPair],
    threshold: float,
    image_count: int,
    vector_count: int,
    skipped: Sequence[str],
    html_pair_limit: int,
    require_identical_alpha: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "similar_pairs.csv"
    json_path = output_dir / "similar_pairs.json"
    html_path = output_dir / "similar_pairs.html"
    groups_csv_path = output_dir / "similar_groups.csv"
    groups_json_path = output_dir / "similar_groups.json"
    groups_html_path = output_dir / "similar_groups.html"
    groups = build_groups(pairs)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "cosine_similarity",
                "first",
                "second",
                "first_variant",
                "second_variant",
                "alpha_identical",
            ]
        )
        for pair in pairs:
            writer.writerow(
                [
                    f"{pair.cosine_similarity:.8f}",
                    pair.first,
                    pair.second,
                    pair.first_variant,
                    pair.second_variant,
                    require_identical_alpha,
                ]
            )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(root),
        "model": MODEL_NAME,
        "threshold": threshold,
        "require_identical_alpha": require_identical_alpha,
        "alpha_comparison": "exact original dimensions and alpha pixels"
        if require_identical_alpha
        else None,
        "image_count": image_count,
        "vector_count": vector_count,
        "pair_count": len(pairs),
        "skipped": list(skipped),
        "pairs": [asdict(pair) for pair in pairs],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with groups_csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "group",
                "image_count",
                "pair_count",
                "minimum_similarity",
                "maximum_similarity",
                "path",
            ]
        )
        for group_number, group in enumerate(groups, start=1):
            for path in group.paths:
                writer.writerow(
                    [
                        group_number,
                        len(group.paths),
                        group.pair_count,
                        f"{group.minimum_similarity:.8f}",
                        f"{group.maximum_similarity:.8f}",
                        path,
                    ]
                )
    groups_json_path.write_text(
        json.dumps(
            {
                "source": str(root),
                "threshold": threshold,
                "require_identical_alpha": require_identical_alpha,
                "alpha_comparison": "exact original dimensions and alpha pixels"
                if require_identical_alpha
                else None,
                "group_count": len(groups),
                "matched_image_count": sum(len(group.paths) for group in groups),
                "groups": [asdict(group) for group in groups],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cards: list[str] = []
    criteria = (
        f"DINOv2 {threshold:.2%}，A 通道尺寸及像素完全一致"
        if require_identical_alpha
        else f"DINOv2 阈值 {threshold:.2%}"
    )
    shown_pairs = pairs[:html_pair_limit]
    for pair in shown_pairs:
        first_path = (root / pair.first).resolve()
        second_path = (root / pair.second).resolve()
        first_url = quote(os.path.relpath(first_path, output_dir).replace("\\", "/"))
        second_url = quote(os.path.relpath(second_path, output_dir).replace("\\", "/"))
        cards.append(
            '<article class="pair">'
            f'<div class="score">{pair.cosine_similarity:.4%}</div>'
            '<div class="images">'
            f'<figure><img loading="lazy" src="{html.escape(first_url)}">'
            f'<figcaption>{html.escape(pair.first)}</figcaption></figure>'
            f'<figure><img loading="lazy" src="{html.escape(second_url)}">'
            f'<figcaption>{html.escape(pair.second)}</figcaption></figure>'
            "</div></article>"
        )
    html_path.write_text(
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DINOv2 相似 PNG</title><style>
body{margin:0;background:#f4f5f7;color:#202124;font:14px system-ui,sans-serif}
header{padding:20px 28px;background:#202124;color:#fff}h1{margin:0 0 6px;font-size:22px}
main{padding:20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}
.pair{background:#fff;border:1px solid #dadce0;border-radius:6px;overflow:hidden}.score{padding:10px 12px;font-weight:700;border-bottom:1px solid #eee}
.images{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#eee}figure{margin:0;background:#fff;min-width:0}
img{display:block;width:100%;height:220px;object-fit:contain;background:#e8eaed}figcaption{padding:9px;overflow-wrap:anywhere;font-size:12px}
</style></head><body>"""
        f"<header><h1>DINOv2 + A 通道相似 PNG</h1><div>{image_count} 张图片，"
        f"{criteria}，找到 {len(pairs)} 对；页面展示前 {len(shown_pairs)} 对，完整结果见 CSV/JSON</div></header>"
        f"<main>{''.join(cards)}</main></body></html>",
        encoding="utf-8",
    )

    group_sections: list[str] = []
    for group_number, group in enumerate(groups, start=1):
        images: list[str] = []
        for path in group.paths:
            image_path = (root / path).resolve()
            image_url = quote(os.path.relpath(image_path, output_dir).replace("\\", "/"))
            images.append(
                f'<figure><img loading="lazy" src="{html.escape(image_url)}">'
                f'<figcaption>{html.escape(path)}</figcaption></figure>'
            )
        group_sections.append(
            '<section class="group">'
            f'<h2>组 {group_number} <span>{len(group.paths)} 张 / {group.pair_count} 对 / '
            f'{group.minimum_similarity:.2%} - {group.maximum_similarity:.2%}</span></h2>'
            f'<div class="group-images">{"".join(images)}</div></section>'
        )
    groups_html_path.write_text(
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DINOv2 相似 PNG 组</title><style>
body{margin:0;background:#f4f5f7;color:#202124;font:14px system-ui,sans-serif}
header{padding:20px 28px;background:#202124;color:#fff}h1{margin:0 0 6px;font-size:22px}
main{padding:20px}.group{margin:0 0 16px;background:#fff;border:1px solid #dadce0;border-radius:6px;overflow:hidden}
h2{margin:0;padding:11px 13px;font-size:16px;border-bottom:1px solid #eee}h2 span{font-size:13px;font-weight:400;color:#5f6368}
.group-images{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:1px;background:#eee}
figure{margin:0;background:#fff;min-width:0}img{display:block;width:100%;height:150px;object-fit:contain;background:#e8eaed}
figcaption{padding:7px;overflow-wrap:anywhere;font-size:11px}
</style></head><body>"""
        f"<header><h1>DINOv2 + A 通道相似 PNG 组</h1><div>{len(groups)} 组，"
        f"{sum(len(group.paths) for group in groups)} 张命中图片，{criteria}</div></header>"
        f"<main>{''.join(group_sections)}</main></body></html>",
        encoding="utf-8",
    )
    return groups_html_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find all PNG pairs above a cosine-similarity threshold using the local DINOv2 ONNX model."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=PROJECT_DIR / ".similar_finder" / "png_pairs"
    )
    parser.add_argument(
        "--model", type=Path, default=PROJECT_DIR / "models" / "dinov2_vits14.onnx"
    )
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--html-pair-limit", type=int, default=1000)
    parser.add_argument(
        "--require-identical-alpha",
        action="store_true",
        help="Only keep pairs whose original dimensions and alpha pixels are identical.",
    )
    parser.add_argument("--rebuild", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        source = args.source.resolve()
        paths = [path for path in discover_images(source) if path.suffix.lower() == ".png"]
        if not paths:
            raise SimilarFinderError(f"no PNG images found below {source}")
        if not 0.0 <= args.threshold <= 1.0:
            raise SimilarFinderError("threshold must be between zero and one")
        model = args.model.resolve()
        manifest = model.with_name("dinov2_vits14_manifest.json")
        print(f"loading local DINOv2 model: {model}", flush=True)
        embedder = OnnxDinoV2Embedder(model, manifest, args.batch_size)
        vectors, entries, skipped = load_or_build_cache(
            source,
            paths,
            args.cache.resolve(),
            model,
            embedder,
            args.rebuild,
            args.progress_every,
        )
        alpha_signatures = None
        if args.require_identical_alpha:
            print("hashing original alpha channels", flush=True)
            alpha_signatures = build_alpha_signatures(source, entries)
        print(f"searching {len(entries)} vectors at >= {args.threshold:.2%}", flush=True)
        pairs = find_pairs(vectors, entries, args.threshold, alpha_signatures)
        report = write_results(
            args.output.resolve(),
            source,
            pairs,
            args.threshold,
            len(paths) - len(skipped),
            len(entries),
            skipped,
            args.html_pair_limit,
            args.require_identical_alpha,
        )
        print(f"found {len(pairs)} image pairs", flush=True)
        print(f"report: {report}", flush=True)
        return 0
    except (OSError, KeyError, json.JSONDecodeError, SimilarFinderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
