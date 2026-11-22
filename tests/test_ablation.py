import json

import pytest

from phasemoe.ablation import load_variants


def test_variant_spec_preserves_controls_and_rejects_ambiguous_names(tmp_path):
    path = tmp_path / "variants.json"
    variants = [
        dict(
            name="grouped_scan",
            strategy="grouped",
            threshold=0,
            kernel_config=dict(packing="triton_scan", combine_n=64),
        )
    ]
    path.write_text(json.dumps(variants))
    loaded = load_variants(path)
    assert loaded[0]["kernel_config"].packing == "triton_scan"
    assert loaded[0]["kernel_config"].combine_n == 64
    for invalid in (
        variants * 2,
        [],
        [dict(name="bad", strategy="typo")],
        [dict(name="bad", threshold=-1)],
        [dict(name="bad", unknown=3)],
    ):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            load_variants(path)
