# PIXELFORGE — Image Processing Studio

A small PySide6 desktop image editor with configurable creative filters.

## Features

- RGB-safe image loading and exporting
- Side-by-side original and edited previews
- Dynamic filter controls driven by `parameters.json`
- Synchronized sliders and spin boxes
- Zoom, fit-to-window, and middle-mouse panning in image viewers
- 29 traditional image filters and effects, including edge detection, blur, and enhancement
- Basic regression tests for filter output contracts

## Requirements

- Python 3.10+
- Packages in `requirements.txt`

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```

Open an image, select a filter from the left panel, adjust its controls, and choose **Export result** to save the preview.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project layout

- `main.py` — application window, image workflow, and filter argument adaptation
- `filters.py` — image processing implementations
- `helper.py` — safe image loading/saving and JSON loading
- `image_viewer.py` — zoomable image display
- `param_ui_parser.py` — JSON-driven parameter UI
- `parameters.json` — filter control definitions
- `tests/` — regression tests

## Notes

- Images are handled internally as contiguous `uint8` RGB NumPy arrays.
- OpenCV uses BGR for file I/O, so conversions happen at the boundaries.
- Parameter names in the UI are normalized to Python-style underscores. Related controls such as RGB tint channels are combined into tuples before calling the relevant filter.


## Expanded filter groups

The filter sidebar is now organized into categories:

- **Creative filters** — existing color and artistic filters.
- **Edge detection and outline effects** — Canny, Sobel, Laplacian, edge overlay, and contour art.
- **Blur effects** — Gaussian, motion, background/bokeh, radial/zoom, tilt-shift, and median blur.
- **Enhancement and analysis** — unsharp-mask sharpening, CLAHE, histogram equalization, thresholding, Fourier frequency filtering, and edge-preserving denoising.

The background/bokeh filter accepts an optional 2-D subject mask when called from code. The UI uses a soft centered elliptical mask as a fallback; it is not semantic subject segmentation. Fourier and edge-analysis filters intentionally return grayscale RGB images to make the detected structure or frequency result easy to inspect.

## Local AI object detection

The **AI Object Detection** entry in the left sidebar supports two local modes:

- **Common objects:** pretrained `yolo26m.pt` for the model's built-in object categories.
- **Open-vocabulary:** pretrained `yolov8l-worldv2.pt`, where you enter categories or descriptions such as `red backpack, wooden chair, helmet`.

Detection runs in a `QThread` so the interface remains responsive. Results include bounding boxes, labels, confidence scores, and an annotated preview that can be exported. All app image data remains RGB `uint8`; conversion to/from OpenCV/Ultralytics BGR happens at the inference boundary. Inference is explicitly set to CPU for broad compatibility.

### Install and prepare pretrained weights

```bash
python -m pip install -r requirements.txt
python download_models.py
python main.py
```

`download_models.py` is an explicit, one-time download step and requires internet access. The running app never downloads model weights and does not upload images. To use the app offline, prepare both files first and keep them under `models/`. You can also browse to model files stored elsewhere. See `models_README.md` for details.

The common detector is suited to its pretrained category set; it does not understand arbitrary text labels. Use open-vocabulary mode for user-specified labels. Open-vocabulary models can still miss objects or return false positives; the confidence threshold and text prompt wording matter. Larger models and higher inference resolution can improve detection in some cases but increase latency and memory use, especially on CPU.

**Licensing:** review Ultralytics licensing terms and the checkpoint's terms before redistributing weights or shipping a commercial application.
