# Fusion Model

Three-input weighted late-fusion model.

Structure

- `fusion/models/` - model definitions
- `fusion/data/` - dataset and transforms
- `fusion/training/` - training and evaluation scripts
- `fusion/utils/` - helper utilities

See notebooks for examples.

Quickstart

1. Create a Python virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note: Installing `torch` may require choosing the correct CUDA / CPU wheel from the official PyTorch instructions at https://pytorch.org.

2. Run a smoke training run (uses synthetic data if no manifest present):

```bash
python3 fusion/training/train.py
```

3. Run unit tests (if `pytest` is installed):

```bash
pytest -q
```

If you don't have `pytest`, you can run the smoke script directly using `python3`.

## BUS-BRA preprocessing

The BUS-BRA training entry point reads `data/busbra/BUSBRA/bus_data.csv` and
requires paired files in `Images/` and `Masks/`. Its default preprocessing follows
the dataset literature: grayscale conversion, a 3x3 median filter, normalization
to `[0, 1]`, lesion bounding-box cropping from the mask, and resizing to 224x224.
Training additionally applies random horizontal/vertical flips, rotations up to
10 degrees, and zoom up to 10 percent. Validation uses the same deterministic
preprocessing without augmentation.

Run a full GPU training job with:

```bash
WANDB_API_KEY="..." WANDB_MODE=online \
python3 training/train.py --device cuda --wandb-project mmfm-busbra
```

See `fusion/notebooks/01_quick_start.ipynb` for examples and dataset manifest format.