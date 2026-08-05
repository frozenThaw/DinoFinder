from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError


APP_VERSION = 1
PREPROCESS_VERSION = 3
DEFAULT_MODEL = "facebook/dinov2-small"
DINOV2_IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ALPHA_BACKGROUNDS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("white", (255, 255, 255)),
    ("gray", (127, 127, 127)),
    ("black", (0, 0, 0)),
)


class SimilarFinderError(RuntimeError):
    """An actionable error that can be shown directly to the user."""


@dataclass(frozen=True)
class VectorEntry:
    path: str
    variant: str


@dataclass(frozen=True)
class Match:
    path: str
    score: float
    source_variant: str
    query_variant: str


@dataclass(frozen=True)
class QueryResult:
    target: str
    matches: list[Match]


def discover_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise SimilarFinderError(f"图片目录不存在: {root}")

    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images, key=lambda path: path.relative_to(root).as_posix().casefold())


def source_fingerprint(root: Path, paths: Sequence[Path], model_name: str) -> str:
    digest = hashlib.sha256()
    settings = {
        "app_version": APP_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "model": model_name,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for path in paths:
        stat = path.stat()
        record = [path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns]
        digest.update(json.dumps(record, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _corner_color(image: Image.Image) -> tuple[int, int, int]:
    rgb = image.convert("RGB")
    width, height = rgb.size
    corners = (
        rgb.getpixel((0, 0)),
        rgb.getpixel((width - 1, 0)),
        rgb.getpixel((0, height - 1)),
        rgb.getpixel((width - 1, height - 1)),
    )
    return tuple(round(sum(pixel[channel] for pixel in corners) / 4) for channel in range(3))


def _square_pad(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    size = max(image.size)
    canvas = Image.new("RGB", (size, size), background)
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.paste(image.convert("RGB"), offset)
    return canvas


def load_image_variants(path: Path) -> list[tuple[str, Image.Image]]:
    """Load an image while preserving all content before DINOv2 center cropping."""
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
    except (OSError, UnidentifiedImageError) as exc:
        raise SimilarFinderError(f"无法读取图片 {path}: {exc}") from exc

    if image.width == 0 or image.height == 0:
        raise SimilarFinderError(f"图片尺寸无效: {path}")

    alpha = image.getchannel("A")
    alpha_min, _ = alpha.getextrema()
    if alpha_min < 255:
        variants: list[tuple[str, Image.Image]] = []
        for name, background in ALPHA_BACKGROUNDS:
            canvas = Image.new("RGBA", image.size, (*background, 255))
            composed = Image.alpha_composite(canvas, image).convert("RGB")
            variants.append((name, _square_pad(composed, background)))
        return variants

    rgb = image.convert("RGB")
    return [("original", _square_pad(rgb, _corner_color(rgb)))]


class DinoV2Embedder:
    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise SimilarFinderError(
                "缺少运行依赖。请先执行 powershell -ExecutionPolicy Bypass -File .\\install.ps1"
            ) from exc

        if batch_size < 1:
            raise SimilarFinderError("batch-size 必须大于 0")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise SimilarFinderError("指定了 CUDA，但当前 PyTorch 未检测到可用的 CUDA 设备")

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size

        print(f"加载 DINOv2 模型 {model_name} ({self.device})...")
        try:
            self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        except Exception as exc:
            raise SimilarFinderError(f"DINOv2 模型加载失败: {exc}") from exc

    def embed(self, images: Sequence[Image.Image]) -> Any:
        import numpy as np

        if not images:
            return np.empty((0, 0), dtype="float32")

        batches: list[Any] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            processed = []
            for image in batch:
                resized = image.convert("RGB").resize(
                    (DINOV2_IMAGE_SIZE, DINOV2_IMAGE_SIZE), Image.Resampling.BICUBIC
                )
                pixels = np.asarray(resized, dtype="float32") / 255.0
                pixels = (pixels - np.asarray(IMAGENET_MEAN, dtype="float32")) / np.asarray(
                    IMAGENET_STD, dtype="float32"
                )
                processed.append(pixels.transpose(2, 0, 1))
            pixel_values = self.torch.from_numpy(np.stack(processed)).to(self.device)
            with self.torch.inference_mode():
                if self.device.type == "cuda":
                    with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                        outputs = self.model(pixel_values=pixel_values)
                else:
                    outputs = self.model(pixel_values=pixel_values)

                features = getattr(outputs, "pooler_output", None)
                if features is None:
                    features = outputs.last_hidden_state[:, 0]
                features = self.torch.nn.functional.normalize(features.float(), p=2, dim=1)
            batches.append(features.cpu().numpy().astype("float32", copy=False))

        return np.concatenate(batches, axis=0)


class SimilarityIndex:
    def __init__(
        self,
        source_dir: Path,
        cache_dir: Path,
        model_name: str,
        embedder: DinoV2Embedder,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise SimilarFinderError(
                "缺少 faiss-cpu。请先执行 powershell -ExecutionPolicy Bypass -File .\\install.ps1"
            ) from exc

        self.faiss = faiss
        self.source_dir = source_dir.resolve()
        self.cache_dir = cache_dir.resolve()
        self.model_name = model_name
        self.embedder = embedder
        self.index: Any | None = None
        self.entries: list[VectorEntry] = []
        self.source_count = 0

    @property
    def index_path(self) -> Path:
        return self.cache_dir / "source.index.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / "source.index.json"

    def ensure(self, rebuild: bool = False) -> bool:
        paths = discover_images(self.source_dir)
        if not paths:
            raise SimilarFinderError(f"源图片目录中没有支持的图片: {self.source_dir}")
        fingerprint = source_fingerprint(self.source_dir, paths, self.model_name)

        if not rebuild and self._load_if_current(fingerprint):
            print(f"复用索引: {self.source_count} 张源图，{len(self.entries)} 个向量")
            return False

        self._build(paths, fingerprint)
        return True

    def _load_if_current(self, fingerprint: str) -> bool:
        if not self.index_path.is_file() or not self.metadata_path.is_file():
            return False
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") != fingerprint:
                return False
            entries = [VectorEntry(**item) for item in metadata["vectors"]]
            index = self.faiss.read_index(str(self.index_path))
            if index.ntotal != len(entries):
                return False
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            return False

        self.entries = entries
        self.index = index
        self.source_count = int(metadata["source_count"])
        return True

    def _build(self, paths: Sequence[Path], fingerprint: str) -> None:
        import numpy as np

        print(f"创建源图库索引: {len(paths)} 张图片...")
        all_vectors: list[Any] = []
        entries: list[VectorEntry] = []
        pending_images: list[Image.Image] = []
        pending_entries: list[VectorEntry] = []
        skipped: list[str] = []

        def flush() -> None:
            if not pending_images:
                return
            all_vectors.append(self.embedder.embed(pending_images))
            entries.extend(pending_entries)
            for image in pending_images:
                image.close()
            pending_images.clear()
            pending_entries.clear()

        flush_limit = max(self.embedder.batch_size * 4, self.embedder.batch_size)
        for number, path in enumerate(paths, start=1):
            try:
                variants = load_image_variants(path)
            except SimilarFinderError as exc:
                skipped.append(str(exc))
                continue
            relative_path = path.relative_to(self.source_dir).as_posix()
            for variant_name, image in variants:
                pending_images.append(image)
                pending_entries.append(VectorEntry(relative_path, variant_name))
            if len(pending_images) >= flush_limit:
                flush()
            if number % 50 == 0 or number == len(paths):
                print(f"  已处理 {number}/{len(paths)}")
        flush()

        if not all_vectors:
            raise SimilarFinderError("没有可用于创建索引的源图片")

        vectors = np.concatenate(all_vectors, axis=0).astype("float32", copy=False)
        self.faiss.normalize_L2(vectors)
        index = self.faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_index = self.index_path.with_suffix(".faiss.tmp")
        temporary_metadata = self.metadata_path.with_suffix(".json.tmp")
        metadata = {
            "app_version": APP_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model_name,
            "source_dir": str(self.source_dir),
            "source_count": len(paths) - len(skipped),
            "vector_count": len(entries),
            "dimension": int(vectors.shape[1]),
            "fingerprint": fingerprint,
            "vectors": [asdict(entry) for entry in entries],
            "skipped": skipped,
        }
        self.faiss.write_index(index, str(temporary_index))
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary_index, self.index_path)
        os.replace(temporary_metadata, self.metadata_path)

        self.index = index
        self.entries = entries
        self.source_count = metadata["source_count"]
        print(f"索引已保存: {self.source_count} 张源图，{len(entries)} 个向量")
        if skipped:
            print(f"警告: 跳过了 {len(skipped)} 张无法读取的图片")

    def search(self, target: Path, top_k: int) -> list[Match]:
        import numpy as np

        if self.index is None:
            raise SimilarFinderError("索引尚未加载")
        if top_k < 1:
            raise SimilarFinderError("top-k 必须大于 0")

        variants = load_image_variants(target)
        variant_names = [name for name, _ in variants]
        images = [image for _, image in variants]
        try:
            queries = self.embedder.embed(images).astype("float32", copy=False)
        finally:
            for image in images:
                image.close()
        self.faiss.normalize_L2(queries)

        # Several vectors may represent the same transparent source image.
        search_k = min(self.index.ntotal, max(top_k * len(ALPHA_BACKGROUNDS) * 4, top_k))
        scores, indices = self.index.search(queries, search_k)
        best_by_path: dict[str, Match] = {}
        for query_number, query_variant in enumerate(variant_names):
            for score, vector_index in zip(scores[query_number], indices[query_number]):
                if vector_index < 0:
                    continue
                entry = self.entries[int(vector_index)]
                absolute_path = str((self.source_dir / entry.path).resolve())
                candidate = Match(
                    path=absolute_path,
                    score=float(np.clip(score, -1.0, 1.0)),
                    source_variant=entry.variant,
                    query_variant=query_variant,
                )
                current = best_by_path.get(entry.path)
                if current is None or candidate.score > current.score:
                    best_by_path[entry.path] = candidate

        return sorted(best_by_path.values(), key=lambda match: match.score, reverse=True)[:top_k]


def write_outputs(
    results: Sequence[QueryResult],
    output_dir: Path,
    model_name: str,
    source_count: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    report_path = output_dir / "report.html"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "source_count": source_count,
        "queries": [
            {"target": result.target, "matches": [asdict(match) for match in result.matches]}
            for result in results
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["target", "rank", "source", "cosine_similarity", "source_variant"])
        for result in results:
            for rank, match in enumerate(result.matches, start=1):
                writer.writerow([result.target, rank, match.path, f"{match.score:.6f}", match.source_variant])

    report_path.write_text(
        _render_report(results, model_name, source_count), encoding="utf-8"
    )
    return report_path


def _image_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def _render_report(
    results: Sequence[QueryResult], model_name: str, source_count: int
) -> str:
    sections: list[str] = []
    for result in results:
        matches_markup: list[str] = []
        for rank, match in enumerate(result.matches, start=1):
            score_percent = max(0.0, min(100.0, match.score * 100.0))
            source_name = Path(match.path).name
            matches_markup.append(
                f"""
                <article class="match">
                  <div class="rank">{rank:02d}</div>
                  <a class="thumb" href="{html.escape(_image_uri(match.path), quote=True)}">
                    <img src="{html.escape(_image_uri(match.path), quote=True)}" alt="{html.escape(source_name, quote=True)}">
                  </a>
                  <div class="match-info">
                    <strong>{html.escape(source_name)}</strong>
                    <span title="{html.escape(match.path, quote=True)}">{html.escape(match.path)}</span>
                  </div>
                  <div class="score">
                    <strong>{match.score:.4f}</strong>
                    <div class="meter"><i style="width:{score_percent:.2f}%"></i></div>
                  </div>
                </article>"""
            )
        target_name = Path(result.target).name
        sections.append(
            f"""
            <section class="query">
              <header class="query-header">
                <a class="target" href="{html.escape(_image_uri(result.target), quote=True)}">
                  <img src="{html.escape(_image_uri(result.target), quote=True)}" alt="{html.escape(target_name, quote=True)}">
                </a>
                <div>
                  <p>目标图片</p>
                  <h2>{html.escape(target_name)}</h2>
                  <span>{html.escape(result.target)}</span>
                </div>
              </header>
              <div class="matches">{''.join(matches_markup)}</div>
            </section>"""
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>相似图片检索结果</title>
  <style>
    :root {{ color-scheme: light; --ink:#171a1d; --muted:#687077; --line:#dfe3e6; --soft:#f5f7f7; --accent:#087f5b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:#fff; font:14px/1.5 "Segoe UI","Microsoft YaHei",sans-serif; letter-spacing:0; }}
    .topbar {{ border-bottom:1px solid var(--line); background:#171a1d; color:#fff; }}
    .topbar-inner, main {{ width:min(1180px, calc(100% - 32px)); margin:0 auto; }}
    .topbar-inner {{ min-height:72px; display:flex; align-items:center; justify-content:space-between; gap:24px; }}
    h1 {{ margin:0; font-size:22px; font-weight:650; }}
    .meta {{ color:#c7cccf; text-align:right; }}
    main {{ padding:28px 0 48px; }}
    .query {{ padding:0 0 34px; margin:0 0 34px; border-bottom:1px solid var(--line); }}
    .query:last-child {{ border-bottom:0; }}
    .query-header {{ display:grid; grid-template-columns:112px minmax(0,1fr); gap:20px; align-items:center; margin-bottom:18px; }}
    .target {{ width:112px; height:112px; display:grid; place-items:center; background:var(--soft); border:1px solid var(--line); }}
    .target img, .thumb img {{ max-width:100%; max-height:100%; object-fit:contain; image-rendering:auto; }}
    .query-header p {{ margin:0 0 4px; color:var(--accent); font-weight:650; }}
    h2 {{ margin:0 0 5px; font-size:18px; overflow-wrap:anywhere; }}
    .query-header span, .match-info span {{ display:block; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .matches {{ border-top:1px solid var(--line); }}
    .match {{ display:grid; grid-template-columns:42px 76px minmax(0,1fr) 120px; gap:14px; align-items:center; min-height:96px; padding:10px 4px; border-bottom:1px solid var(--line); }}
    .rank {{ color:#8a9298; font-variant-numeric:tabular-nums; font-weight:650; }}
    .thumb {{ width:76px; height:76px; display:grid; place-items:center; background:var(--soft); border:1px solid var(--line); }}
    .match-info {{ min-width:0; }}
    .match-info strong {{ display:block; margin-bottom:3px; overflow-wrap:anywhere; }}
    .score {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .score strong {{ font-size:16px; }}
    .meter {{ height:4px; margin-top:8px; background:#e8ebed; overflow:hidden; }}
    .meter i {{ display:block; height:100%; background:var(--accent); }}
    @media (max-width:700px) {{
      .topbar-inner {{ align-items:flex-start; flex-direction:column; justify-content:center; padding:14px 0; gap:3px; }}
      .meta {{ text-align:left; }}
      .query-header {{ grid-template-columns:84px minmax(0,1fr); gap:14px; }}
      .target {{ width:84px; height:84px; }}
      .match {{ grid-template-columns:28px 62px minmax(0,1fr); gap:10px; }}
      .thumb {{ width:62px; height:62px; }}
      .score {{ grid-column:3; text-align:left; width:100%; }}
    }}
  </style>
</head>
<body>
  <div class="topbar"><div class="topbar-inner">
    <h1>相似图片检索结果</h1>
    <div class="meta">DINOv2 + FAISS · {source_count} 张源图 · {html.escape(model_name)}</div>
  </div></div>
  <main>{''.join(sections)}</main>
</body>
</html>
"""


def copy_best_matches(results: Sequence[QueryResult], output_dir: Path) -> None:
    destination = output_dir / "best_matches"
    destination.mkdir(parents=True, exist_ok=True)
    for result in results:
        if not result.matches:
            continue
        target_stem = Path(result.target).stem
        source = Path(result.matches[0].path)
        output_name = f"{target_stem}__MATCH__{source.name}"
        shutil.copy2(source, destination / output_name)


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="使用 DINOv2 特征和 FAISS 从源图库中查找目标图片。"
    )
    parser.add_argument("--source", type=Path, default=project_dir / "SourceImage", help="源图库目录")
    parser.add_argument("--targets", type=Path, default=project_dir / "TargetImage", help="目标图片目录")
    parser.add_argument("--output", type=Path, default=project_dir / "Output", help="结果输出目录")
    parser.add_argument("--cache", type=Path, default=project_dir / ".similar_finder", help="FAISS 索引缓存目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face DINOv2 模型名称或本地路径")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--batch-size", type=int, default=16, help="特征提取批大小")
    parser.add_argument("--top-k", type=int, default=5, help="每张目标图返回的结果数")
    parser.add_argument("--rebuild", action="store_true", help="忽略已有缓存并重建源图库索引")
    parser.add_argument("--copy-best", action="store_true", help="把每张目标图的最佳匹配复制到输出目录")
    return parser


def run(args: argparse.Namespace) -> int:
    source_dir = args.source.resolve()
    target_dir = args.targets.resolve()
    targets = discover_images(target_dir)
    if not targets:
        raise SimilarFinderError(f"目标图片目录中没有支持的图片: {target_dir}")

    embedder = DinoV2Embedder(args.model, args.device, args.batch_size)
    similarity_index = SimilarityIndex(source_dir, args.cache, args.model, embedder)
    similarity_index.ensure(rebuild=args.rebuild)

    results: list[QueryResult] = []
    for number, target in enumerate(targets, start=1):
        print(f"检索目标 [{number}/{len(targets)}]: {target.name}")
        matches = similarity_index.search(target, args.top_k)
        results.append(QueryResult(str(target.resolve()), matches))
        for rank, match in enumerate(matches, start=1):
            print(f"  {rank:>2}. {match.score:.4f}  {match.path}")

    report_path = write_outputs(results, args.output.resolve(), args.model, similarity_index.source_count)
    if args.copy_best:
        copy_best_matches(results, args.output.resolve())
    print(f"完成。可视化报告: {report_path}")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except KeyboardInterrupt:
        print("操作已取消", file=sys.stderr)
        return 130
    except SimilarFinderError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
