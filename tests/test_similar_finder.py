import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from similar_finder import (
    Match,
    QueryResult,
    SimilarityIndex,
    VectorEntry,
    _render_report,
    discover_images,
    load_image_variants,
    source_fingerprint,
)


class SimilarFinderTests(unittest.TestCase):
    def test_discover_images_ignores_meta_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (8, 8)).save(root / "z.PNG")
            Image.new("RGB", (8, 8)).save(root / "a.png")
            (root / "a.png.meta").write_text("metadata", encoding="utf-8")

            self.assertEqual([path.name for path in discover_images(root)], ["a.png", "z.PNG"])

    def test_transparent_image_has_three_square_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sprite.png"
            image = Image.new("RGBA", (24, 8), (255, 0, 0, 0))
            image.putpixel((12, 4), (255, 0, 0, 255))
            image.save(path)

            variants = load_image_variants(path)
            self.assertEqual([name for name, _ in variants], ["white", "gray", "black"])
            self.assertTrue(all(image.size == (24, 24) for _, image in variants))
            for _, variant in variants:
                variant.close()

    def test_fingerprint_changes_with_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "image.png"
            Image.new("RGB", (8, 8)).save(path)

            small = source_fingerprint(root, [path], "facebook/dinov2-small")
            base = source_fingerprint(root, [path], "facebook/dinov2-base")
            self.assertNotEqual(small, base)

    def test_search_deduplicates_background_variants(self) -> None:
        class FakeFaiss:
            @staticmethod
            def normalize_L2(vectors) -> None:
                return None

        class FakeIndex:
            ntotal = 3

            @staticmethod
            def search(queries, count):
                return (
                    np.array([[0.70, 0.90, 0.80]], dtype="float32"),
                    np.array([[0, 1, 2]], dtype="int64"),
                )

        class FakeEmbedder:
            @staticmethod
            def embed(images):
                return np.array([[1.0, 0.0]], dtype="float32")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.png"
            Image.new("RGB", (8, 8)).save(target)

            finder = SimilarityIndex.__new__(SimilarityIndex)
            finder.faiss = FakeFaiss()
            finder.source_dir = root
            finder.embedder = FakeEmbedder()
            finder.index = FakeIndex()
            finder.entries = [
                VectorEntry("same.png", "white"),
                VectorEntry("same.png", "black"),
                VectorEntry("other.png", "original"),
            ]

            matches = finder.search(target, top_k=2)
            self.assertEqual([Path(match.path).name for match in matches], ["same.png", "other.png"])
            self.assertAlmostEqual(matches[0].score, 0.9, places=6)
            self.assertEqual(matches[0].source_variant, "black")

    def test_report_escapes_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target & query.png"
            source = root / "source & best.png"
            Image.new("RGB", (8, 8)).save(target)
            Image.new("RGB", (8, 8)).save(source)
            result = QueryResult(
                str(target),
                [Match(str(source), 0.75, "original", "original")],
            )

            rendered = _render_report([result], "facebook/dinov2-small", 1)
            self.assertIn("target &amp; query.png", rendered)
            self.assertIn("source &amp; best.png", rendered)
            self.assertNotIn("source & best.png", rendered)


if __name__ == "__main__":
    unittest.main()
