import torch

from experiment_data import SEQUENCE_FEATURES
from model_fixture import load_demo_batch, make_demo_model


def test_irrigation_gradient_matches_central_difference():
    sequence, static, planting, capacity, initial = load_demo_batch(count=2)
    model = make_demo_model(sequence, static).double().eval()
    sequence = sequence.double().requires_grad_(True)
    static = static.double()
    capacity = capacity.double()
    initial = initial.double()
    irrigation_index = SEQUENCE_FEATURES.index("IRR")
    day = 90
    predicted = model(sequence, static, planting, capacity, initial)["yield"].sum()
    automatic = torch.autograd.grad(predicted, sequence)[0][0, day, irrigation_index]

    epsilon = 1e-3
    plus = sequence.detach().clone()
    minus = sequence.detach().clone()
    plus[0, day, irrigation_index] += epsilon
    minus[0, day, irrigation_index] -= epsilon
    with torch.no_grad():
        y_plus = model(plus, static, planting, capacity, initial)["yield"][0]
        y_minus = model(minus, static, planting, capacity, initial)["yield"][0]
    numerical = (y_plus - y_minus) / (2.0 * epsilon)
    assert torch.isfinite(automatic)
    assert torch.isfinite(numerical)
    assert torch.allclose(automatic, numerical, rtol=2e-2, atol=2e-3)
