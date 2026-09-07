# Fusion Model

Three-input weighted late-fusion model.

Structure

- `models/` - model definitions
- `data/` - dataset and transforms
- `training/` - training and evaluation scripts
- `utils/` - helper utilities

See notebooks for examples.

Quickstart

1. Create a Python virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Note: Installing `torch` may require choosing the correct CUDA / CPU wheel from the official PyTorch instructions at https://pytorch.org.

2. Run a training job (defaults to the BUS-BRA dataset; requires the corresponding `datasets/` checkout — see the BUS-BRA section below):

```bash
python3 training/train.py --device cpu
```

3. Run unit tests (if `pytest` is installed):

```bash
pytest -q
```

If you don't have `pytest`, you can run the smoke script directly using `python3`.

## BUS-BRA preprocessing

The BUS-BRA training entry point reads `datasets/ultrasound/busbra/BUSBRA/bus_data.csv` and
requires paired files in `Images/` and `Masks/`. Its default preprocessing follows
the dataset literature: grayscale conversion, a 3x3 median filter, normalization
to `[0, 1]`, lesion bounding-box cropping from the mask, and resizing to 224x224.
Training additionally applies random horizontal/vertical flips, rotations up to
10 degrees, and zoom up to 10 percent. Validation uses the same deterministic
preprocessing without augmentation. Single-mode paper-reproduction runs do not
use modality dropout; that augmentation is reserved for a later fusion experiment.

Run a full GPU training job with:

```bash
WANDB_API_KEY="..." WANDB_MODE=online \
python3 training/train.py --single-mode --device cuda --wandb-project mmfm-busbra
```

Training defaults to CUDA and raises an error when no GPU is available. Use
`--device cpu` explicitly for local CPU smoke tests.

The MIAS lesion-patch path can be selected with `--dataset mias`; it uses the
coordinates and radii in `Info.txt` and writes `mias_single_<backbone>_best.pth`.
The local BUSI subset can be selected with `--dataset busi`; it reads labels from
`datasets/ultrasound/BUS/BUS/DatasetB.xlsx` and uses paired files in `original/`
and `GT/`. This checkout contains 163 images, not the complete 780-image release.
The BrEaST classification path can be selected with `--dataset breast`; it reads
`Classification` labels from the clinical workbook and uses `*_tumor.png` files as
lesion masks. The `*_other*` files are not diagnosis labels.
The BreaDM MRI classification path can be selected with `--dataset breamdm`; it
uses the official nine-channel `img9Se` arrays and their train/validation splits.

See `notebooks/01_quick_start.ipynb` for examples and dataset manifest format.