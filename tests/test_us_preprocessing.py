import numpy as np
import pytest
import torch
from PIL import Image

from data.us_preprocessing import (
    Pipeline, PreprocessedTransform, _fuzzy_cmeans_1d, add_synthetic_markers, build_pipeline, hist_match,
    hist_spec, histogram_of, intensity_window, line_removal, parse_step,
)
from models.cyclegan import (
    BottleneckClassifier, NCIAGenerator, PatchDiscriminator, ResnetGenerator, load_generator,
)


def _speckle(seed=0, shape=(64, 80)):
    return np.random.default_rng(seed).integers(20, 200, shape).astype(np.uint8)


@pytest.mark.parametrize("spec", [
    "hist_eq", "hist_spec", "intensity_window:low=1;high=99", "line_removal:line_length=2;min_area=3",
    "cross_removal",
])
def test_every_step_keeps_shape_and_dtype(spec):
    image = _speckle()
    out = np.asarray(Pipeline([spec])(Image.fromarray(image)))

    assert out.shape == image.shape and out.dtype == np.uint8


def test_hist_match_to_own_histogram_is_identity_and_follows_reference():
    image, reference = _speckle(0), np.full((10, 10), 200, dtype=np.uint8)

    assert np.array_equal(hist_match(image, histogram_of([image])), image)
    assert (hist_match(image, reference) == 200).all()


def test_intensity_window_stretches_percentile_range():
    image = np.linspace(50, 150, 101, dtype=np.float64).round().astype(np.uint8).reshape(1, -1)

    out = intensity_window(image, low=0, high=100)

    assert out.min() == 0 and out.max() == 255


def test_fuzzy_cmeans_finds_separated_intensity_groups():
    levels = np.arange(256, dtype=np.float64)
    weights = np.zeros(256)
    weights[[20, 120, 230]] = 1000

    centres, membership = _fuzzy_cmeans_1d(levels, weights)

    assert np.allclose(sorted(centres), [20, 120, 230], atol=2)
    assert np.allclose(membership.sum(axis=0), 1)


def test_hist_spec_keeps_dark_regions_darker_than_bright_ones():
    rng = np.random.default_rng(0)
    image = np.clip(np.concatenate([rng.normal(50, 15, (20, 40)), rng.normal(180, 15, (20, 40))]), 0, 255).astype(np.uint8)

    out = hist_spec(image)

    assert out[:20].mean() < out[20:].mean()


def test_line_removal_inpaints_a_thin_bright_line():
    image = np.full((40, 40), 80, dtype=np.uint8)
    image[20, 5:35] = 250

    out = line_removal(image, line_length=4, min_area=5)

    assert out[20, 5:35].mean() < 150


def test_parse_step_and_unknown_step():
    assert parse_step("perc_hm:reference=r.npy;low=1") == ("perc_hm", {"reference": "r.npy", "low": "1"})
    with pytest.raises(ValueError):
        parse_step("sharpen")
    assert build_pipeline(None) is None


def test_preprocessed_transform_caches_by_source_path(tmp_path):
    path = tmp_path / "img.png"
    Image.fromarray(_speckle()).save(path)
    calls = []
    transform = PreprocessedTransform(Pipeline(["hist_eq"]), lambda image: calls.append(image) or image,
                                      cache_dir=tmp_path / "cache")

    first = np.asarray(transform(Image.open(path)))
    second = np.asarray(transform(Image.open(path)))

    assert len(list((tmp_path / "cache").rglob("*.png"))) == 1
    assert np.array_equal(first, second) and len(calls) == 2


def test_synthetic_markers_only_add_bright_annotations():
    image = np.full((200, 200), 60, dtype=np.uint8)
    mask = np.zeros_like(image)
    mask[90:110, 90:110] = 255

    marked = add_synthetic_markers(image, mask, np.random.default_rng(3))

    changed = marked != image
    assert changed.any() and (marked[changed] > image[changed]).all()


def test_generators_and_discriminator_shapes():
    x = torch.randn(1, 1, 64, 64)
    resnet, ncia = ResnetGenerator(n_blocks=2), NCIAGenerator()

    assert resnet(x).shape == x.shape and resnet.encode(x).shape == (1, 256, 16, 16)
    out, aux = ncia(x, return_aux=True)
    assert out.shape == x.shape and aux.shape == (1, 50) and ncia.info_feature(x).shape == (1, 256)
    assert PatchDiscriminator()(x).shape == (1, 1, 6, 6)
    assert BottleneckClassifier((256, 16, 16), 2)(resnet.encode(x)).shape == (1, 2)


def test_load_generator_reads_junyanz_dataparallel_and_mmfm_checkpoints(tmp_path):
    junyanz = {f"module.{k}": v for k, v in ResnetGenerator(3, 3).state_dict().items()}
    torch.save(junyanz, tmp_path / "latest_net_G_A.pth")
    ncia = NCIAGenerator()
    torch.save({"variant": "ncia", "input_nc": 1, "G_A2B": ncia.state_dict(), "G_B2A": ncia.state_dict()},
               tmp_path / "mmfm.pth")

    generator, variant = load_generator(tmp_path / "latest_net_G_A.pth")
    assert variant == "cyclegan" and generator.input_nc == 3
    generator, variant = load_generator(tmp_path / "mmfm.pth", direction="B2A")
    assert variant == "ncia" and not generator.training
