import pandas as pd
import pytest

pytest.importorskip("torch")
from src import data  # noqa: E402


@pytest.mark.parametrize("name,expected", [
    ("person12_bacteria_34.jpeg", "P12"),
    ("person12_virus_35.jpeg", "P12"),          # same patient, different image
    ("IM-0115-0001.jpeg", "N-0115"),
    ("IM-0115-0002.jpeg", "N-0115"),
    ("NORMAL2-IM-0373-0001.jpeg", "N2-0373"),
    ("weird_name.jpeg", "F-weird_name"),         # unparseable -> its own patient
])
def test_patient_id(name, expected):
    assert data.patient_id(name) == expected


def _make_tree(root):
    """Tiny fake dataset: several images per patient, like the real one."""
    def touch(split, cls, name):
        d = root / split / cls
        d.mkdir(parents=True, exist_ok=True)
        (d / name).touch()

    for split, normal_ids, pneu_ids in [
        ("train", range(1, 61), range(1, 151)),
        ("val", range(61, 63), range(151, 153)),
        ("test", range(200, 220), range(300, 340)),
    ]:
        for i in normal_ids:
            for k in range(1, 3):
                touch(split, "NORMAL", f"IM-{i:04d}-{k:04d}.jpeg")
        for i in pneu_ids:
            for k in range(1, 4):
                touch(split, "PNEUMONIA", f"person{i}_bacteria_{k}.jpeg")
    return root


@pytest.fixture()
def df(tmp_path):
    return data.scan_dataset(_make_tree(tmp_path))


def test_scan_dataset(df):
    assert set(df.columns) >= {"path", "label", "y", "orig_split", "patient"}
    assert set(df.label) == {"NORMAL", "PNEUMONIA"}
    assert (df[df.label == "PNEUMONIA"].y == 1).all()
    assert set(df.orig_split) == {"train", "val", "test"}


def test_scan_dataset_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.scan_dataset(tmp_path)


def test_splits_have_no_patient_leakage(df):
    out = data.make_splits(df)
    data.check_no_leakage(out)  # raises AssertionError on leakage


def test_original_test_is_kept_as_external(df):
    out = data.make_splits(df)
    assert (out[out.orig_split == "test"].split == "external").all()
    assert not (out[out.orig_split != "test"].split == "external").any()


def test_split_proportions_and_classes(df):
    out = data.make_splits(df)
    pool = out[out.split != "external"]
    frac = pool.split.value_counts(normalize=True)
    assert 0.60 <= frac["train"] <= 0.80
    assert 0.08 <= frac["val"] <= 0.22
    assert 0.08 <= frac["test"] <= 0.22
    for s in ("train", "val", "test"):
        assert set(out[out.split == s].label) == {"NORMAL", "PNEUMONIA"}


def test_splits_are_reproducible(df):
    a = data.make_splits(df, seed=42).split
    b = data.make_splits(df, seed=42).split
    assert a.equals(b)


def test_leakage_check_catches_a_leak(df):
    out = data.make_splits(df)
    victim = out[out.split == "train"].patient.iloc[0]
    out.loc[out[out.split == "val"].index[0], "patient"] = victim
    with pytest.raises(AssertionError):
        data.check_no_leakage(out)


def test_class_weights_and_pos_weight():
    d = pd.DataFrame({"y": [0] * 25 + [1] * 75})
    w = data.class_weights(d)
    assert w[0] > w[1]                      # minority class weighs more
    assert abs(float(w.mean()) - 1.0) < 1e-6
    assert abs(float(data.pos_weight(d)) - 25 / 75) < 1e-6
