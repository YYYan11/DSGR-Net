import torch

from model_fixture import load_demo_batch, make_demo_model


def test_hard_recurrence_closes_daily_water_balance():
    sequence, static, planting, capacity, initial = load_demo_batch()
    model = make_demo_model(sequence, static).eval()
    with torch.no_grad():
        output = model(sequence, static, planting, capacity, initial)
    closure = output["water_closure_error"]
    assert torch.max(torch.abs(closure)).item() < 1e-5
