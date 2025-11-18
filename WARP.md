# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

This repository contains a single-page Streamlit application (`streamlit_app.py`) that loads a pre-trained TensorFlow model to classify fruit images as **fresh** or **spoiled**. The app provides both file upload and camera input options, performs basic image preprocessing, runs the model to obtain a prediction, and displays the result with a styled dark UI.

The TensorFlow model is expected to be available as `fruit_fresh_spoiled_model.h5` in the project root.

## Environment & Dependencies

- Python dependencies are pinned in `requirements.txt`. This list is large and includes many libraries that are not used directly in `streamlit_app.py`.
- The app itself only relies on:
  - `streamlit`
  - `tensorflow`
  - `numpy`
  - `Pillow`

### Recommended setup

From the repo root:

- Create and activate a virtual environment (PowerShell / Windows):
  - `python -m venv .venv`
  - `.\.venv\Scripts\Activate.ps1`
- Install dependencies:
  - To install everything exactly as captured: `pip install -r requirements.txt`
  - To install just the core runtime dependencies, you may instead choose a minimal set, e.g.:
    - `pip install streamlit tensorflow numpy pillow`

Ensure `fruit_fresh_spoiled_model.h5` is present in the repository root or update `MODEL_PATH` in `streamlit_app.py` accordingly.

## Common Commands

All commands are intended to be run from the repo root.

### Run the Streamlit app

- Default run:
  - `streamlit run streamlit_app.py`

If you are running multiple Streamlit apps or need a specific port, you can pass `--server.port` and `--server.address`, e.g.:
- `streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0`

### Linting

`ruff` is included in `requirements.txt`, but no configuration file is present.

- Lint the main app file with default settings:
  - `ruff streamlit_app.py`

If you add more Python modules, you can lint them by passing their paths or the entire directory, e.g. `ruff .`.

### Testing

`pytest` is included in `requirements.txt`, but there are currently **no test files** in this repository.

If you add tests (e.g. under a `tests/` directory or following `test_*.py` / `*_test.py` naming), you can run them with:
- `pytest`

To run a single test module once tests exist:
- `pytest path/to/test_file.py`

## Code Architecture & Structure

Currently, the application logic is concentrated in `streamlit_app.py`. The key conceptual pieces are:

1. **Styling & Layout (lines ~6–113, 145–154)**
   - Custom CSS is injected via `st.markdown(..., unsafe_allow_html=True)` to create a dark theme and style headers, buttons, upload/camera widgets, and image display.
   - The main Streamlit layout is defined with a title, description, and a radio control to select the image input source.

2. **Model Loading (lines ~115–123)**
   - `MODEL_PATH` points to `fruit_fresh_spoiled_model.h5` in the repo root.
   - `load_my_model()` uses `st.cache_resource` to load the TensorFlow model once per process and reuse it across requests, which is critical for performance.
   - The loaded model is stored in a module-level variable `model` and reused by the classification function.

3. **Inference & Preprocessing (lines ~125–143)**
   - `IMG_SIZE` defines the target resize dimensions `(150, 150)`.
   - `classify_fruit_streamlit(img)` handles:
     - Converting NumPy arrays to `PIL.Image` when needed.
     - Converting to RGB, resizing, normalizing pixel values to `[0, 1]`, and adding a batch dimension.
     - Calling `model.predict` and interpreting the scalar output as a binary classification (threshold at `0.5`).
   - The function returns a textual label (with emoji) and a confidence score for use in the UI.

4. **Input Handling (lines ~149–168)**
   - A radio selector chooses between:
     - `Upload Picture` (file uploader for JPEG/PNG images), and
     - `Camera Input` (camera capture via `st.camera_input`).
   - Depending on the choice, the corresponding image source is used to produce a `PIL.Image` instance (`image_to_process`).

5. **Result Display (lines ~170–180)**
   - If an image is provided, it is displayed with `st.image` and then passed to `classify_fruit_streamlit`.
   - The prediction is rendered using:
     - `st.success` for "Fresh" predictions.
     - `st.error` for "Spoiled" predictions.
   - Confidence is shown as a percentage.

## Guidelines for Future Changes

- **Adding new input sources**: Follow the existing pattern of:
  - Extending the `input_source` `st.radio` options.
  - Preparing an image object (`PIL.Image` or NumPy array) and assigning it to `image_to_process`.
  - Reusing `classify_fruit_streamlit` so preprocessing and inference remain centralized.

- **Changing the model**:
  - Update `MODEL_PATH` to point to the new `.h5` (or SavedModel) and ensure the model’s expected input size and output semantics match what `classify_fruit_streamlit` assumes.
  - If the new model expects a different input size or returns multi-class probabilities, adjust `IMG_SIZE`, the preprocessing logic, and the label/threshold logic accordingly.

- **Refactoring**:
  - As the app grows, consider splitting logic into separate modules (e.g. `model.py` for loading/inference and `ui.py` for layout) and updating imports in `streamlit_app.py`.
