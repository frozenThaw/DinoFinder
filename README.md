# SimilarFinder

SimilarFinder is a native Windows image-search application built with C++20, Qt 6, OpenCV, DINOv2 through ONNX Runtime CUDA, and FAISS HNSW. The packaged application does not require Python, PyTorch, or a locally installed CUDA Toolkit.

## Application workflow

1. Start `SimilarFinder.exe`.
2. Select `Choose Directory`, then select `Build` to create the catalog.
3. Select `Choose Image` or drag an image into the window.
4. Adjust `Similar Value` and `Top Num` as needed.
5. Double-click a result to open it, or right-click to show it in Explorer or copy its path.

The default similarity threshold is `0.80`. Only results with cosine similarity greater than or equal to that value are displayed. `Top Num` defaults to `200` and is an upper limit; the application does not add lower-scoring results to fill the list.

Catalog data is stored below `%LOCALAPPDATA%\SimilarFinder\catalogs`. Source directories remain read-only. Rebuilding the same directory performs an incremental sync and detects new, modified, and deleted files. Interrupted builds resume from their latest durable checkpoint.

## Runtime behavior

- Supported inputs: PNG, JPG/JPEG, BMP, WebP, and TIFF.
- Transparent images generate white, gray, and black background embeddings.
- Images are square-padded, resized to 224 x 224, normalized with ImageNet parameters, and embedded as 384-dimensional DINOv2-small CLS vectors.
- ONNX Runtime uses the CUDA provider when available and falls back to CPU when CUDA initialization or inference cannot continue.
- FAISS uses normalized inner-product HNSW search with stable 64-bit vector IDs.
- Catalog updates become inactive through feature generations. The index and FP16 feature log are compacted after inactive vectors reach 15 percent.

## Build prerequisites

- Windows 10 or 11 x64 with an AVX2-capable CPU
- Visual Studio with the Desktop C++ and CMake workloads
- Python is needed only by the dependency bootstrap and optional ONNX export tooling; it is not used by the application or portable package

From PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_dependencies.ps1 `
  -Python C:\Path\To\python.exe

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1
```

The bootstrap installs Qt 6.8.3 and ONNX Runtime GPU 1.22 under `.deps`, and uses vcpkg for OpenCV, FAISS, Eigen, and their native dependencies.

The pinned ONNX model is stored at `models\dinov2_vits14.onnx`; its metadata and SHA-256 are stored beside it in `dinov2_vits14_manifest.json`.

To regenerate and numerically validate the development model artifact:

```powershell
python -m pip install -r .\tools\model_export\requirements.txt
python .\tools\model_export\export_dinov2.py
```

## Portable package

Download the pinned NVIDIA redistributables, then build the package:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap_cuda_redist.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\package.ps1
```

The CUDA bootstrap downloads official NVIDIA CUDA 12.8.1, cuBLAS 12.8, cuFFT 11.3, and cuDNN 9.8 archives and validates their SHA-256 values. `CUDA_REDIST_ROOT` can override the generated `.deps\cuda-redist\bin` directory.

The package script builds and tests Release mode, runs `windeployqt`, copies ONNX Runtime, vcpkg and NVIDIA runtime DLLs, installs the ONNX model, emits SHA-256 checksums, and creates `dist\SimilarFinder-windows-x64.zip`. Python files and environments are not copied.

## Project layout

- `src/core`: OpenCV preprocessing, ONNX inference, persistent catalog, FP16 feature log, and FAISS HNSW search.
- `src/app`: Qt Widgets UI, background service coordination, asynchronous thumbnails, and the result grid.
- `tests/cpp`: native preprocessing, storage, filtering, ID, and FAISS tests.
- `tools/model_export`: development-only ONNX export and PyTorch parity validation.
- `scripts`: dependency bootstrap, native build/test, and portable packaging.

The earlier `similar_finder.py` command-line prototype remains as a development reference and is not included in the native package.
