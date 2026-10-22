import numpy as np
import pandas as pd
import torch
from PIL import Image

from data.dataset import PadToSquareTransform, ShenTransform
from training.train import _fixed_test_kfold_frame, _mean_gray, _patient_split


def _frame(num_patients=20, rows_per_patient=3):
    rows = []
    for patient in range(num_patients):
        for _ in range(rows_per_patient):
            rows.append({"patient": f"P{patient}", "label": patient % 2,
                         "split": "test" if patient >= num_patients - 4 else "trainval"})
    return pd.DataFrame(rows)


def test_patient_split_keeps_each_patient_on_one_side():
    train, validation = _patient_split(_frame(), "patient", seed=0, validation_fraction=0.25)

    assert not set(train.patient) & set(validation.patient)
    assert len(train) + len(validation) == len(_frame())
    assert validation.patient.nunique() == 5


def test_fixed_test_kfold_keeps_official_test_and_rotates_validation():
    metadata = _frame()
    validation_patients = []
    for fold in range(1, 5):
        train, validation, test = _fixed_test_kfold_frame(metadata, "patient", 4, fold, seed=0)
        assert set(test.patient) == {"P16", "P17", "P18", "P19"}
        assert not set(train.patient) & set(validation.patient)
        assert not (set(train.patient) | set(validation.patient)) & set(test.patient)
        validation_patients.append(set(validation.patient))

    assert set().union(*validation_patients) == {f"P{i}" for i in range(16)}


def test_pad_to_square_preserves_aspect_with_zero_padding():
    image = Image.fromarray(np.full((10, 40), 255, dtype=np.uint8), mode="L")

    output = PadToSquareTransform(size=40)(image)

    assert output.shape == (3, 40, 40)
    assert output[0, 0, 0] == 0 and output[0, 20, 20] == 1.0


def test_shen_transform_subtracts_mean_and_augments_to_fixed_shape():
    image = Image.fromarray(np.full((30, 20), 102, dtype=np.uint8), mode="L")

    plain = ShenTransform(size=16, mean=0.4)(image)
    augmented = ShenTransform(size=16, augment=True, mean=0.4)(image)

    assert plain.shape == augmented.shape == (3, 16, 16)
    assert torch.isclose(plain[0, 8, 8], torch.tensor(0.0), atol=1e-6)


def test_mean_gray_is_pixel_weighted(tmp_path):
    small, large = tmp_path / "small.png", tmp_path / "large.png"
    Image.fromarray(np.zeros((1, 1), dtype=np.uint8)).save(small)
    Image.fromarray(np.full((3, 1), 255, dtype=np.uint8)).save(large)

    assert _mean_gray([small, large]) == 0.75


def test_bus_uclm_drops_normals_and_merges_chco_into_chvi(tmp_path):
    from training.train import _bus_uclm_metadata

    pd.DataFrame({'Image': ['CHCO_000.png', 'CHVI_005.png', 'ALWI_001.png', 'ALWI_002.png'],
                  'Label': ['Benign', 'Benign ', 'Malignant', 'Normal']}).to_csv(tmp_path / 'INFO.csv', sep=';', index=False)

    metadata = _bus_uclm_metadata(tmp_path)

    assert list(metadata.label) == [0, 0, 1]
    assert list(metadata.patient) == ['CHVI', 'CHVI', 'ALWI']


def test_dmid_mirrors_right_breasts_and_skips_duplicates(tmp_path, monkeypatch):
    import types
    import pydicom
    import audit.sources
    from audit.sample import Sample
    from training.train import _dmid_metadata

    image = np.zeros((400, 600), dtype=np.uint8)
    image[:, 300:] = 200            # breast on the right
    image[100:150, 500:550] = 255   # lesion

    def sample(sample_id, label):
        return Sample(sample_id=sample_id, dataset='dmid', modality='mammography', label_name='x', label=label,
                      used=True, group=sample_id, split=None, fold=None, native_kind='dicom',
                      native_path='unused.dcm', box=(500, 100, 550, 150), extra={'view': 'CC', 'roi_mask': 'ok'})

    monkeypatch.setattr(audit.sources, 'dmid', lambda root: [sample('IMG001', 1), sample('IMG319', 0)])
    monkeypatch.setattr(pydicom, 'dcmread', lambda path: types.SimpleNamespace(pixel_array=image))

    metadata = _dmid_metadata(tmp_path, cache_dir=tmp_path / 'cache', margin=0.2, min_side=60)

    assert list(metadata.patient) == ['IMG001'] and metadata.side[0] == 'R'
    crop = np.asarray(Image.open(metadata.image[0]))
    assert crop.shape == (70, 70)
    assert crop[10:60, 10:60].min() == 255   # lesion centred after mirroring
