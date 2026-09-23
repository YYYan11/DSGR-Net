import torch

from model_fixture import load_demo_batch, make_demo_model


def test_demo_forward_is_finite_and_has_expected_shapes():
    sequence, static, planting, capacity, initial = load_demo_batch()
    model = make_demo_model(sequence, static).eval()
    with torch.no_grad():
        output = model(sequence, static, planting, capacity, initial)
    assert output["yield"].shape == (len(sequence),)
    assert output["soil_water"].shape[:2] == sequence.shape[:2]
    assert output["process_state_sequence"].shape[:2] == sequence.shape[:2]
    for key in ("yield", "soil_water", "actual_et", "growth", "water_stress"):
        assert torch.isfinite(output[key]).all(), key
